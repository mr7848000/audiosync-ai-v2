const form = document.getElementById("form");
const videoInput = document.getElementById("video");
const audioInput = document.getElementById("audio");
const submit = document.getElementById("submit");

videoInput.addEventListener("change", () => {
  document.getElementById("vname").textContent = videoInput.files[0]?.name || "Choose a video file";
});
audioInput.addEventListener("change", () => {
  document.getElementById("aname").textContent = audioInput.files[0]?.name || "Choose Hindi audio";
});

function showError(message) {
  const box = document.getElementById("error");
  box.textContent = message;
  box.classList.remove("hidden");
}
function setProgress(value, message) {
  const n = Math.max(0, Math.min(100, Number(value) || 0));
  document.getElementById("bar").style.width = n + "%";
  document.getElementById("percent").textContent = n + "%";
  document.getElementById("message").textContent = message || "Working...";
}
async function readResponse(response) {
  const raw = await response.text();
  let data;
  try { data = JSON.parse(raw); }
  catch {
    const snippet = raw.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim().slice(0, 220);
    throw new Error(`Server returned a non-JSON response (HTTP ${response.status}). ${snippet || "The service may have restarted or is temporarily unavailable."}`);
  }
  if (!response.ok) throw new Error(data.detail || data.message || `Server error: ${response.status}`);
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  document.getElementById("error").classList.add("hidden");
  document.getElementById("result").classList.add("hidden");
  if (!videoInput.files[0] || !audioInput.files[0]) {
    showError("Please select both the original video and Hindi audio.");
    return;
  }
  const data = new FormData();
  data.append("video", videoInput.files[0]);
  data.append("hindi_audio", audioInput.files[0]);
  submit.disabled = true;
  submit.textContent = "Uploading files...";
  document.getElementById("progress").classList.remove("hidden");
  setProgress(2, "Uploading files. Keep this page open.");
  try {
    const response = await fetch("/api/process", { method: "POST", body: data });
    const job = await readResponse(response);
    document.getElementById("title").textContent = "Processing your video...";
    await poll(job.job_id);
  } catch (error) {
    showError(error.message || "Something went wrong.");
  } finally {
    submit.disabled = false;
    submit.innerHTML = 'Generate Hindi Video <span>→</span>';
  }
});

async function poll(jobId) {
  while (true) {
    let response;
    try {
      response = await fetch(`/api/status/${encodeURIComponent(jobId)}`);
      const job = await readResponse(response);
      setProgress(job.progress, job.message);
      if (job.status === "completed") {
        document.getElementById("info").textContent =
          `${job.sentence_count || "Your"} sentence groups processed. Original audio removed.`;
        document.getElementById("download").href = job.download_url;
        document.getElementById("result").classList.remove("hidden");
        document.getElementById("title").textContent = "Completed";
        return;
      }
      if (job.status === "failed") throw new Error(job.message || "Processing failed.");
    } catch (error) {
      throw new Error(error.message || "The server became unavailable. Check Render and try again.");
    }
    await new Promise(resolve => setTimeout(resolve, 4000));
  }
}
