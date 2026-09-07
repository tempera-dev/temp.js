//! Storage for an explicitly selected, trusted local application directory.
//!
//! Reject redirected or shared journal files before SQLite can modify them.
//! This is not a sandbox against a hostile same-UID process renaming ancestors
//! concurrently. Application roots must not be writable by untrusted actors.
//! SQLite's no-follow open is additional protection, not a directory capability.

use std::path::Path;

use anyhow::Result;
use rusqlite::Connection;

#[cfg(not(unix))]
pub(super) fn open_connection(_app_dir: &Path) -> Result<Connection> {
    anyhow::bail!("private journal storage requires a supported Unix filesystem")
}

#[cfg(unix)]
pub(super) fn open_connection(app_dir: &Path) -> Result<Connection> {
    use std::fs::{self, DirBuilder, OpenOptions, Permissions};
    use std::io::ErrorKind;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};

    use anyhow::{Context, ensure};
    use rusqlite::OpenFlags;

    // Resolve only the caller-selected root. Never canonicalize a child, since
    // doing so would turn an untrusted child symlink into an accepted target.
    let root = app_dir.canonicalize().context("app root must exist")?;
    ensure!(root.is_dir(), "app root must be a directory");
    let dir = root.join(".beater");
    match DirBuilder::new().mode(0o700).create(&dir) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error).context("create private journal directory"),
    }
    // Open the directory itself without following a final symlink; chmod its
    // handle, never a potentially redirected pathname. Legacy directory modes
    // can be tightened safely without opening any database inode.
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(&dir)
        .context("journal directory must be a real directory, not a symlink")?;
    let directory_metadata = directory.metadata()?;
    ensure!(directory_metadata.is_dir(), "invalid journal directory");
    directory.set_permissions(Permissions::from_mode(0o700))?;

    // Include rollback journals: SQLite may encounter one before switching an
    // existing database to WAL. Validate *all* existing members before creating
    // or opening the database. Sidecars may legitimately be absent.
    let members = [
        "journal.db",
        "journal.db-wal",
        "journal.db-shm",
        "journal.db-journal",
    ];
    let validate = |path: &Path| -> Result<()> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error).context("inspect journal member"),
        };
        ensure!(
            metadata.file_type().is_file(),
            "journal members must be regular non-symlink files"
        );
        ensure!(
            metadata.nlink() == 1,
            "journal members must have exactly one link"
        );
        ensure!(
            metadata.uid() == directory_metadata.uid(),
            "journal member owner differs from storage directory"
        );
        // Do not open/chmod an existing SQLite file: closing an extra fd can
        // release this process's POSIX SQLite locks. Refuse a legacy permissive
        // file instead; the operator must tighten it with all runtimes stopped.
        ensure!(
            metadata.mode() & 0o077 == 0,
            "journal members require private permissions; stop runtimes and set owner-only file permissions"
        );
        Ok(())
    };
    for name in members {
        validate(&dir.join(name))?;
    }

    let database = dir.join("journal.db");
    // Atomic creation gives a new database mode 0600 before SQLite sees it.
    // create_new never opens an existing inode, preserving other connections'
    // POSIX locks. It also rejects symlinks, including dangling symlinks.
    match OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&database)
    {
        Ok(file) => drop(file),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error).context("create private journal database"),
    }
    for name in members {
        validate(&dir.join(name))?;
    }
    // No CREATE (already privately created), no URI interpretation, and no
    // symlink resolution. Bundled SQLite's Unix VFS also no-follows sidecars.
    Connection::open_with_flags(
        database,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )
    .context("open private journal database")
}

#[cfg(all(test, unix))]
mod tests {
    use std::fs;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::path::PathBuf;

    use crate::Journal;

    struct Root(PathBuf);

    impl Root {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "beater-private-journal-{}-{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Root {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn redirected_directory_does_not_create_an_outside_database() {
        let app = Root::new();
        let outside = Root::new();
        fs::write(outside.0.join("canary"), b"untouched").unwrap();
        let before = fs::metadata(&outside.0).unwrap().permissions().mode();
        symlink(&outside.0, app.0.join(".beater")).unwrap();
        assert!(Journal::open(&app.0).is_err());
        assert!(!outside.0.join("journal.db").exists());
        assert_eq!(fs::read(outside.0.join("canary")).unwrap(), b"untouched");
        assert_eq!(
            fs::metadata(&outside.0).unwrap().permissions().mode(),
            before
        );
    }

    #[test]
    fn symlinked_database_and_all_sidecars_leave_outside_files_untouched() {
        for name in [
            "journal.db",
            "journal.db-wal",
            "journal.db-shm",
            "journal.db-journal",
        ] {
            let app = Root::new();
            let outside = Root::new();
            let dir = app.0.join(".beater");
            fs::create_dir(&dir).unwrap();
            let target = outside.0.join("outside.db");
            let conn = rusqlite::Connection::open(&target).unwrap();
            conn.execute_batch(
                "CREATE TABLE canary(value TEXT); INSERT INTO canary VALUES ('untouched');",
            )
            .unwrap();
            drop(conn);
            let before = fs::read(&target).unwrap();
            let before_mode = fs::metadata(&target).unwrap().permissions().mode();
            symlink(&target, dir.join(name)).unwrap();
            assert!(Journal::open(&app.0).is_err(), "accepted {name}");
            assert_eq!(fs::read(&target).unwrap(), before, "modified {name}");
            assert_eq!(
                fs::metadata(&target).unwrap().permissions().mode(),
                before_mode
            );
            if name != "journal.db" {
                assert!(
                    !dir.join("journal.db").exists(),
                    "created DB before validating {name}"
                );
            }
        }
    }

    #[test]
    fn dangling_symlinks_and_non_file_members_are_rejected() {
        for name in [
            "journal.db",
            "journal.db-wal",
            "journal.db-shm",
            "journal.db-journal",
        ] {
            let app = Root::new();
            let dir = app.0.join(".beater");
            fs::create_dir(&dir).unwrap();
            let target = app.0.join("must-not-be-created");
            symlink(&target, dir.join(name)).unwrap();
            assert!(Journal::open(&app.0).is_err());
            assert!(!target.exists());
            fs::remove_file(dir.join(name)).unwrap();
            fs::create_dir(dir.join(name)).unwrap();
            assert!(Journal::open(&app.0).is_err());
        }
    }

    #[test]
    fn hardlinked_database_and_sidecars_are_rejected_before_mutation() {
        for name in [
            "journal.db",
            "journal.db-wal",
            "journal.db-shm",
            "journal.db-journal",
        ] {
            let app = Root::new();
            let outside = Root::new();
            let dir = app.0.join(".beater");
            fs::create_dir(&dir).unwrap();
            let target = outside.0.join("canary");
            fs::write(&target, b"untouched").unwrap();
            fs::set_permissions(&target, fs::Permissions::from_mode(0o600)).unwrap();
            fs::hard_link(&target, dir.join(name)).unwrap();
            assert!(Journal::open(&app.0).is_err());
            assert_eq!(fs::read(&target).unwrap(), b"untouched");
        }
    }

    #[test]
    fn new_storage_is_private_and_existing_connections_still_work() {
        let app = Root::new();
        let first = Journal::open(&app.0).unwrap();
        first.create_run("one", "agent", "input").unwrap();
        let second = Journal::open(&app.0).unwrap();
        second.create_run("two", "agent", "input").unwrap();
        assert_eq!(first.run("two").unwrap().id, "two");
        let dir = app.0.join(".beater");
        assert_eq!(
            fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        for name in ["journal.db", "journal.db-wal", "journal.db-shm"] {
            assert_eq!(
                fs::metadata(dir.join(name)).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        drop(second);
        drop(first);
        let reopened = Journal::open(&app.0).unwrap();
        assert_eq!(reopened.run("one").unwrap().id, "one");
    }

    #[test]
    fn permissive_legacy_files_require_explicit_offline_migration() {
        let app = Root::new();
        drop(Journal::open(&app.0).unwrap());
        let database = app.0.join(".beater/journal.db");
        fs::set_permissions(&database, fs::Permissions::from_mode(0o644)).unwrap();
        let before = fs::read(&database).unwrap();
        let error = Journal::open(&app.0).err().unwrap().to_string();
        assert!(error.contains("stop runtimes"));
        assert_eq!(fs::read(&database).unwrap(), before);
        fs::set_permissions(&database, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(Journal::open(&app.0).is_ok());
    }

    #[test]
    fn selected_existing_root_is_resolved_but_missing_or_file_roots_are_rejected() {
        let parent = Root::new();
        let selected = parent.0.join("selected");
        fs::create_dir(&selected).unwrap();
        let alias = parent.0.join("explicit-root-alias");
        symlink(&selected, &alias).unwrap();
        assert!(Journal::open(&alias).is_ok());
        assert!(selected.join(".beater/journal.db").exists());
        assert!(Journal::open(&selected.join("..").join("selected")).is_ok());
        let missing = parent.0.join("missing");
        assert!(Journal::open(&missing).is_err());
        assert!(!missing.exists());
        let file = parent.0.join("file");
        fs::write(&file, b"untouched").unwrap();
        assert!(Journal::open(&file).is_err());
        assert_eq!(fs::read(&file).unwrap(), b"untouched");
    }
}
