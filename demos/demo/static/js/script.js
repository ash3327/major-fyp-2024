let includeAlphabets = true;
let isWebcamMode = true;
let updateInterval;

// Function to fetch and update stored gestures
function fetchStoredGestures() {
  fetch("/get_stored_gestures")
    .then((response) => response.json())
    .then((gestures) => {
      const container = document.getElementById("stored-gestures");
      container.innerHTML = ""; // Clear existing gestures

      if (gestures.length === 0) {
        container.innerHTML =
          '<p class="text-muted">No stored gestures yet. Press spacebar to store a gesture!</p>';
        return;
      }

      gestures.forEach((gesture) => {
        const card = document.createElement("div");
        card.className = "gesture-card";
        card.innerHTML = `
                            <img src="${gesture.image_path}" alt="${gesture.name}" class="gesture-image">
                            <p class="gesture-name">${gesture.name}</p>
                        `;
        container.appendChild(card);
      });
    })
    .catch((error) => console.error("Error fetching stored gestures:", error));
}

// Function to fetch and update the gesture classification result
function fetchPrediction() {
  fetch("/prediction")
    .then((response) => response.json())
    .then((data) => {
      document.getElementById(
        "currentPrediction"
      ).textContent = `Current Prediction: ${data.prediction}`;

      // Update prediction rankings
      const rankingsContainer = document.getElementById("predictionRankings");
      rankingsContainer.innerHTML = ""; // Clear previous rankings

      if (data.all_predictions && data.all_predictions.length > 0) {
        // Find the minimum distance for normalization
        const minDistance = Math.min(
          ...data.all_predictions.map((p) => p.distance)
        );

        // Display top 5 predictions
        data.all_predictions.slice(0, 5).forEach((pred) => {
          const container = document.createElement("div");
          container.className = "prediction-bar-container";

          const label = document.createElement("span");
          label.className = "prediction-label";
          label.textContent = pred.class;

          const barContainer = document.createElement("div");
          barContainer.style.display = "inline-block";
          barContainer.style.width = "200px";

          const bar = document.createElement("div");
          bar.className = "prediction-bar";
          // Calculate width based on relative distance (inverse relationship)
          const width = Math.max(20, (minDistance / pred.distance) * 100);
          bar.style.width = `${width}%`;

          const distance = document.createElement("span");
          distance.className = "prediction-distance";
          distance.textContent = pred.distance.toFixed(3);

          barContainer.appendChild(bar);
          container.appendChild(label);
          container.appendChild(barContainer);
          container.appendChild(distance);
          rankingsContainer.appendChild(container);
        });
      }
    })
    .catch((error) => console.error("Error fetching prediction:", error));
}

// Function to update PCA plot
function updatePCAPlot() {
  fetch("/pca_data")
    .then((response) => response.json())
    .then((data) => {
      const traces = [];

      // Original class means
      traces.push({
        x: data.means.x,
        y: data.means.y,
        text: data.means.labels,
        mode: "markers+text",
        type: "scatter",
        marker: {
          size: 10,
          color: "blue",
        },
        textposition: "top center",
        name: "Original Classes",
      });

      // Custom class means if they exist
      if (data.custom_means) {
        traces.push({
          x: data.custom_means.x,
          y: data.custom_means.y,
          text: data.custom_means.labels,
          mode: "markers+text",
          type: "scatter",
          marker: {
            size: 10,
            color: "green",
          },
          textposition: "top center",
          name: "Custom Classes",
        });
      }

      // Current gesture point if it exists
      if (data.current) {
        traces.push({
          x: [data.current.x],
          y: [data.current.y],
          mode: "markers",
          type: "scatter",
          marker: {
            size: 12,
            color: "red",
            symbol: "star",
          },
          name: "Current Gesture",
        });
      }

      const layout = {
        title: "PCA Visualization",
        xaxis: { title: "First Principal Component" },
        yaxis: { title: "Second Principal Component" },
        showlegend: true,
        legend: {
          x: 1,
          xanchor: "right",
          y: 1,
        },
      };

      Plotly.newPlot("pca-plot", traces, layout);
    })
    .catch((error) => console.error("Error fetching PCA data:", error));
}

// Function to store current gesture
function storeGesture() {
  const container = document.getElementById("stored-gestures");
  const tempCard = document.createElement("div");
  tempCard.className = "gesture-card saving";
  tempCard.innerHTML = "<p>Saving gesture...</p>";
  container.insertBefore(tempCard, container.firstChild);

  fetch("/store_gesture", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        console.error("Error storing gesture:", data.error);
        tempCard.remove();
        showSaveIndicator("Failed to save gesture");
        window.location.reload();
        return;
      }

      // Remove the temporary card
      tempCard.remove();

      // Create the new gesture card with animation
      const newCard = document.createElement("div");
      newCard.className = "gesture-card new";
      newCard.innerHTML = `
                    <img src="${data.image_path}" alt="${data.gesture_name}" class="gesture-image">
                    <p class="gesture-name">${data.gesture_name}</p>
                `;
      container.insertBefore(newCard, container.firstChild);

      showSaveIndicator("Gesture saved successfully!");
    })
    .catch((error) => {
      // console.error('Error storing gesture:', error);
      // tempCard.remove();
      // showSaveIndicator('Failed to save gesture');
      window.location.reload();
    });
}

// Function to toggle alphabet classification
function toggleAlphabets() {
  fetch("/toggle_alphabets", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      includeAlphabets = data.include_alphabets;
      const button = document.getElementById("toggle-alphabets");
      button.textContent = includeAlphabets
        ? "Disable Alphabets"
        : "Enable Alphabets";
      button.className = includeAlphabets
        ? "btn btn-danger"
        : "btn btn-success";
    })
    .catch((error) => console.error("Error toggling alphabets:", error));
}

// Function to toggle between webcam and image upload modes
function toggleMode() {
  isWebcamMode = !isWebcamMode;
  const button = document.getElementById("toggle-mode");
  const uploadControls = document.getElementById("upload-controls");
  const webcamFeed = document.getElementById("webcam-feed");
  const imageDisplay = document.getElementById("image-display");

  if (isWebcamMode) {
    button.textContent = "Switch to Image";
    uploadControls.style.display = "none";
    webcamFeed.style.display = "block";
    imageDisplay.style.display = "none";
    // Restart periodic updates
    startPeriodicUpdates();
  } else {
    button.textContent = "Switch to Webcam";
    uploadControls.style.display = "block";
    webcamFeed.style.display = "none";
    imageDisplay.style.display = "block";
    // Stop periodic updates
    stopPeriodicUpdates();
  }

  // Call backend to sync mode
  fetch("/toggle_mode", {
    method: "POST",
  });
}

// Function to upload and process image
function uploadImage() {
  const fileInput = document.getElementById("image-upload");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image first");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  fetch("/upload_image", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        alert(data.error);
        return;
      }

      // Update the image display
      document.getElementById("uploaded-image").src = data.image;

      // Update prediction and rankings
      document.getElementById(
        "currentPrediction"
      ).textContent = `Current Prediction: ${data.prediction}`;

      // Update prediction rankings
      const rankingsContainer = document.getElementById("predictionRankings");
      rankingsContainer.innerHTML = ""; // Clear previous rankings

      if (data.all_predictions && data.all_predictions.length > 0) {
        // Find the minimum distance for normalization
        const minDistance = Math.min(
          ...data.all_predictions.map((p) => p.distance)
        );

        // Display top 5 predictions
        data.all_predictions.slice(0, 5).forEach((pred) => {
          const container = document.createElement("div");
          container.className = "prediction-bar-container";

          const label = document.createElement("span");
          label.className = "prediction-label";
          label.textContent = pred.class;

          const barContainer = document.createElement("div");
          barContainer.style.display = "inline-block";
          barContainer.style.width = "200px";

          const bar = document.createElement("div");
          bar.className = "prediction-bar";
          // Calculate width based on relative distance (inverse relationship)
          const width = Math.max(20, (minDistance / pred.distance) * 100);
          bar.style.width = `${width}%`;

          const distance = document.createElement("span");
          distance.className = "prediction-distance";
          distance.textContent = pred.distance.toFixed(3);

          barContainer.appendChild(bar);
          container.appendChild(label);
          container.appendChild(barContainer);
          container.appendChild(distance);
          rankingsContainer.appendChild(container);
        });
      }

      // Update PCA plot
      updatePCAPlot();
    })
    .catch((error) => {
      console.error("Error uploading image:", error);
      alert("Error uploading image. Please try again.");
    });
}

// Function to start periodic updates
function startPeriodicUpdates() {
  updateInterval = setInterval(() => {
    if (isWebcamMode) {
      fetchPrediction();
      updatePCAPlot();
    }
  }, 500);
}

// Function to stop periodic updates
function stopPeriodicUpdates() {
  if (updateInterval) {
    clearInterval(updateInterval);
  }
}

// Function to show save indicator
function showSaveIndicator(message) {
  const indicator = document.getElementById("save-indicator");
  indicator.textContent = message;
  indicator.style.display = "block";
  setTimeout(() => {
    indicator.style.display = "none";
  }, 2000);
}

// Initialize the page
document.addEventListener("DOMContentLoaded", function () {
  fetchStoredGestures();
  startPeriodicUpdates();

  // Update model info in the UI
  fetch("/model_info")
    .then((response) => response.json())
    .then((data) => {
      document.querySelector(".alert-info strong").textContent =
        data.model_name;
    })
    .catch((error) => console.error("Error fetching model info:", error));
});

// Handle keyboard events
document.addEventListener("keydown", function (event) {
  if (event.code === "Space") {
    event.preventDefault(); // Prevent page scroll
    storeGesture();
  }
});
