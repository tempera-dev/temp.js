//! Exclusive run ownership for a trusted local app directory.
//!
//! Keep lock files permanently: unlinking permits two owners of different inodes
//! at the same path. OS process death or File drop releases the advisory lock.
//! This is not a distributed lease or a hostile shared-filesystem boundary.
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::Path;

pub(crate) struct RunOwnership {
    _file: File,
}

impl RunOwnership {
    pub(crate) fn acquire(app_dir: &Path, run_id: &str) -> io::Result<Self> {
        if run_id.is_empty()
            || run_id.len() > 128
            || !run_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid run identity",
            ));
        }
        let root = app_dir.canonicalize()?;
        if !root.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "app root must be a directory",
            ));
        }
        let journal_dir = root.join(".beater");
        ensure_directory(&journal_dir)?;
        let lock_dir = journal_dir.join("run-locks");
        ensure_directory(&lock_dir)?;
        let path = lock_dir.join(format!("{run_id}.lock"));
        match fs::symlink_metadata(&path) {
            Ok(meta) if !meta.file_type().is_file() => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "run lock must be a regular file",
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if file.metadata()?.nlink() != 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "run lock must have one link",
                ));
            }
        }
        file.try_lock()
            .map_err(|error| io::Error::other(format!("run ownership unavailable: {error}")))?;
        Ok(Self { _file: file })
    }
}

fn ensure_directory(path: &Path) -> io::Result<()> {
    match fs::create_dir(path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    if !fs::symlink_metadata(path)?.file_type().is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "run directory must not be a symlink",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::RunOwnership;
    use std::fs;
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    struct TempRoot(PathBuf);
    impl TempRoot {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir()
                .join(format!("tempera-run-lock-{}-{unique}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    struct ChildGuard(Child);
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    fn child(root: &TempRoot, mode: &str) -> ChildGuard {
        ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    &format!(
                        "{}::child_process",
                        module_path!().split_once("::").unwrap().1
                    ),
                    "--nocapture",
                ])
                .env("TEMPERA_LOCK_TEST_ROOT", &root.0)
                .env("TEMPERA_LOCK_TEST_MODE", mode)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        )
    }
    #[test]
    fn child_process() {
        let Some(root) = std::env::var_os("TEMPERA_LOCK_TEST_ROOT") else {
            return;
        };
        let root = PathBuf::from(root);
        let mode = std::env::var("TEMPERA_LOCK_TEST_MODE").unwrap();
        let ownership = RunOwnership::acquire(&root, "run-1");
        if mode == "contend" {
            assert!(ownership.is_err());
            return;
        }
        let _ownership = ownership.unwrap();
        if mode == "hold" {
            fs::write(root.join("ready"), b"ready").unwrap();
            std::thread::sleep(Duration::from_secs(30));
        }
    }
    #[test]
    fn same_run_exclusive_other_runs_independent_inode_retained() {
        let root = TempRoot::new();
        let owner = RunOwnership::acquire(&root.0, "run-1").unwrap();
        assert!(RunOwnership::acquire(&root.0, "run-1").is_err());
        let _other = RunOwnership::acquire(&root.0, "run-2").unwrap();
        let path = root.0.join(".beater/run-locks/run-1.lock");
        #[cfg(unix)]
        let inode = {
            use std::os::unix::fs::MetadataExt;
            fs::metadata(&path).unwrap().ino()
        };
        drop(owner);
        assert!(path.exists());
        let _next = RunOwnership::acquire(&root.0, "run-1").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(inode, fs::metadata(path).unwrap().ino());
        }
    }
    #[test]
    fn actual_process_contention_and_crash_release() {
        let root = TempRoot::new();
        let mut owner = child(&root, "hold");
        let deadline = Instant::now() + Duration::from_secs(5);
        while !root.0.join("ready").exists() {
            assert!(owner.0.try_wait().unwrap().is_none());
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(child(&root, "contend").0.wait().unwrap().success());
        assert!(RunOwnership::acquire(&root.0, "run-1").is_err());
        owner.0.kill().unwrap();
        owner.0.wait().unwrap();
        assert!(child(&root, "acquire").0.wait().unwrap().success());
        assert!(root.0.join(".beater/run-locks/run-1.lock").exists());
    }
    #[test]
    fn invalid_identity_cannot_create_journal_paths() {
        let root = TempRoot::new();
        for id in ["", "../escape", "a/b", "a.b", "é", &"x".repeat(129)] {
            assert!(RunOwnership::acquire(&root.0, id).is_err());
        }
        assert!(!root.0.join(".beater").exists());
    }
    #[test]
    #[cfg(unix)]
    fn symlink_directory_and_lock_file_rejected() {
        use std::os::unix::fs::symlink;
        let root = TempRoot::new();
        let target = TempRoot::new();
        symlink(&target.0, root.0.join(".beater")).unwrap();
        assert!(RunOwnership::acquire(&root.0, "run-1").is_err());
        fs::remove_file(root.0.join(".beater")).unwrap();
        drop(RunOwnership::acquire(&root.0, "run-1").unwrap());
        let lock = root.0.join(".beater/run-locks/run-2.lock");
        fs::write(target.0.join("file"), b"").unwrap();
        symlink(target.0.join("file"), lock).unwrap();
        assert!(RunOwnership::acquire(&root.0, "run-2").is_err());
    }
}
