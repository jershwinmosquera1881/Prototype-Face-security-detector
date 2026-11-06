const cameraCard = document.getElementById("cameraCard");
    const video = document.getElementById("cameraVideo");
    const scanButton = document.getElementById("scanButton");
    const buttonText = document.getElementById("buttonText");
    const statusTitle = document.getElementById("statusTitle");
    const statusDetail = document.getElementById("statusDetail");
    const faceCount = document.getElementById("faceCount");
    const scanStatus = document.getElementById("scanStatus");
    const notice = document.getElementById("notice");
    let stream = null;
    let detector = null;
    let detectorKind = null;
    let detectorPromise = null;
    let detectionTimer = null;
    let noticeTimer = null;
    let detectionInProgress = false;
    let scanning = false;
    let scanGeneration = 0;

    const faceModelUrl = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
    const visionWasmUrl = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm";
    const visionModuleUrl = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision_bundle.mjs";

    const clockTime = document.getElementById("clockTime");
    const clockDate = document.getElementById("clockDate");

    function updateClock() {
      const now = new Date();
      clockTime.textContent = new Intl.DateTimeFormat("en-PH", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
      }).format(now);
      clockDate.textContent = new Intl.DateTimeFormat("en-PH", {
        weekday: "long",
        year: "numeric",
        month: "long",
        day: "numeric"
      }).format(now);
    }

    updateClock();
    window.setInterval(updateClock, 1000);

    function showNotice(message) {
      notice.textContent = message;
      notice.classList.add("show");
      window.clearTimeout(noticeTimer);
      noticeTimer = window.setTimeout(() => notice.classList.remove("show"), 3600);
    }

    function setStatus(state, title, detail) {
      scanStatus.dataset.state = state;
      statusTitle.textContent = title;
      statusDetail.textContent = detail;
    }

    function updateFaces(count) {
      faceCount.textContent = `${count} ${count === 1 ? "face" : "faces"}`;
      if (count > 0) {
        setStatus("scanning", count === 1 ? "Face detected" : `${count} faces detected`, "Scanning continues.");
      } else {
        setStatus("scanning", "Looking for faces", "Center your face in the frame.");
      }
    }

    async function initializeDetector() {
      if (detector) return detector;
      if (detectorPromise) return detectorPromise;

      detectorPromise = (async () => {
        let visionError;
        try {
          const { FilesetResolver, FaceDetector } = await import(visionModuleUrl);
          const vision = await FilesetResolver.forVisionTasks(visionWasmUrl);
          try {
            detector = await FaceDetector.createFromOptions(vision, {
              baseOptions: { modelAssetPath: faceModelUrl, delegate: "GPU" },
              runningMode: "VIDEO",
              minDetectionConfidence: 0.5
            });
          } catch {
            detector = await FaceDetector.createFromOptions(vision, {
              baseOptions: { modelAssetPath: faceModelUrl, delegate: "CPU" },
              runningMode: "VIDEO",
              minDetectionConfidence: 0.5
            });
          }
          detectorKind = "mediapipe";
          return detector;
        } catch (error) {
          visionError = error;
        }

        if ("FaceDetector" in window) {
          try {
            detector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 10 });
            detectorKind = "native";
            return detector;
          } catch (error) {
            throw new Error(`Face detection is unavailable. ${error.message || visionError.message || ""}`.trim());
          }
        }

        throw new Error(`Face detection could not load. Check your internet connection and browser support. ${visionError.message || ""}`.trim());
      })();

      try {
        return await detectorPromise;
      } finally {
        detectorPromise = null;
      }
    }

    async function detectFaces() {
      if (!detector || !scanning || detectionInProgress || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      detectionInProgress = true;
      try {
        const activeDetector = detector;
        const result = detectorKind === "mediapipe"
          ? activeDetector.detectForVideo(video, performance.now())
          : await activeDetector.detect(video);
        if (scanning && detector === activeDetector) updateFaces(result.detections ? result.detections.length : result.length);
      } catch (error) {
        stopScan();
        setStatus("error", "Scan stopped", "The camera frame could not be read. Try again.");
        showNotice(`Could not read the camera frame: ${error.message || "unknown error."}`);
      } finally {
        detectionInProgress = false;
      }
    }

    function stopScan(state = "stopped") {
      scanGeneration += 1;
      scanning = false;
      window.clearInterval(detectionTimer);
      detectionTimer = null;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }
      video.srcObject = null;
      cameraCard.classList.remove("camera-on");
      cameraCard.classList.remove("detecting");
      buttonText.textContent = "Start scan";
      scanButton.setAttribute("aria-label", "Start face scan");
      scanButton.disabled = false;
      setStatus(state, "Camera paused", "Select the button to resume.");
      faceCount.textContent = "—";
    }

    async function startScan() {
      const activeScan = scanGeneration;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        buttonText.textContent = "Try again";
        scanButton.setAttribute("aria-label", "Try opening the camera again");
        scanButton.disabled = false;
        setStatus("error", "Camera unavailable", "Open this page over HTTPS or localhost.");
        showNotice("Camera access requires HTTPS or localhost.");
        return;
      }
      scanButton.disabled = true;
      buttonText.textContent = "Connecting…";
      setStatus("starting", "Requesting permission", "Allow camera access in your browser.");
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: "user", width: { ideal: 720 }, height: { ideal: 960 } }
        });
        if (activeScan !== scanGeneration) {
          stream.getTracks().forEach(track => track.stop());
          stream = null;
          return;
        }
        video.srcObject = stream;
        await video.play();
        if (activeScan !== scanGeneration) {
          stream.getTracks().forEach(track => track.stop());
          stream = null;
          video.srcObject = null;
          return;
        }
        cameraCard.classList.add("camera-on");
        scanning = true;
        scanButton.disabled = false;
        buttonText.textContent = "Stop camera";
        scanButton.setAttribute("aria-label", "Stop camera");

        setStatus("starting", "Loading face detector", "Preparing on-device detection. First load needs an internet connection.");
        try {
          await initializeDetector();
          if (!scanning || activeScan !== scanGeneration) return;
          cameraCard.classList.add("detecting");
          setStatus("scanning", "Looking for faces", "Center your face in the frame.");
          buttonText.textContent = "Stop scan";
          scanButton.setAttribute("aria-label", "Stop face scan");
          detectionTimer = window.setInterval(detectFaces, 300);
          detectFaces();
        } catch (error) {
          if (!scanning || activeScan !== scanGeneration) return;
          setStatus("unsupported", "Camera preview only", error.message);
          faceCount.textContent = "—";
          showNotice("Face detection is unavailable. Check your internet connection or use a supported browser.");
        }
      } catch (error) {
        if (stream) {
          stream.getTracks().forEach(track => track.stop());
          stream = null;
        }
        video.srcObject = null;
        cameraCard.classList.remove("camera-on", "detecting");
        scanning = false;
        scanButton.disabled = false;
        buttonText.textContent = "Try again";
        scanButton.setAttribute("aria-label", "Try opening the camera again");
        const permissionDenied = error.name === "NotAllowedError" || error.name === "PermissionDeniedError";
        const noCamera = error.name === "NotFoundError" || error.name === "DevicesNotFoundError";
        const detail = permissionDenied
          ? "Allow camera access in your browser settings, then try again."
          : noCamera
            ? "No camera was found on this device."
            : "Check camera access and connection, then try again.";
        setStatus("error", "Camera unavailable", detail);
        showNotice(`Could not start the camera: ${error.message || "access was denied or unavailable."}`);
      }
    }

    scanButton.addEventListener("click", () => {
      if (scanning) stopScan();
      else startScan();
    });

    document.getElementById("helpButton").addEventListener("click", () => {
      showNotice("Camera access requires your permission. Video is not saved. Face detection may not be supported by this browser.");
    });

    window.addEventListener("pagehide", () => {
      stopScan();
      if (detector && typeof detector.close === "function") detector.close();
      detector = null;
      detectorKind = null;
    });

    startScan();
