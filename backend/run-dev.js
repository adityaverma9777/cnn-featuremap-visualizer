const { spawn } = require("child_process");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
const requirementsPath = path.join(repoRoot, "requirements.txt");

function runCommand(command, args, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: "inherit",
      shell: false,
    });

    child.on("error", (error) => {
      reject(error);
    });
    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${command} exited with code ${code}`));
      }
    });
  });
}

async function detectPython() {
  const candidates = ["py", "python", "python3"];

  for (const cmd of candidates) {
    try {
      await runCommand(cmd, ["--version"], repoRoot);
      return cmd;
    } catch {
      // Try next candidate.
    }
  }

  throw new Error("Python was not found in PATH. Install Python first.");
}

async function main() {
  const python = await detectPython();

  let dependenciesReady = true;
  try {
    await runCommand(
      python,
      ["-c", "import fastapi, uvicorn, torch, torchvision, PIL, multipart"],
      repoRoot
    );
  } catch {
    dependenciesReady = false;
  }

  if (!dependenciesReady) {
    console.log("Installing backend dependencies...");
    await runCommand(python, ["-m", "pip", "install", "-r", requirementsPath], repoRoot);
  }

  console.log("Starting backend on http://127.0.0.1:8000");
  await runCommand(
    python,
    ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
    repoRoot
  );
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
