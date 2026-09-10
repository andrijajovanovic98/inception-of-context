/**
 * Inception-of-Context (IoC) — Part 1 Presentation Deck Engine
 * Handles keyboard navigation, fullscreen, slide counter, and overview grid.
 * Zero external dependencies. 100% offline.
 */

(function () {
  "use strict";

  const slides = document.querySelectorAll(".slide");
  const totalSlides = slides.length;
  let currentIndex = 0;

  const counterEl = document.getElementById("slideCounter");
  const progressBar = document.getElementById("progressBar");
  const btnPrev = document.getElementById("btnPrev");
  const btnNext = document.getElementById("btnNext");
  const btnFullscreen = document.getElementById("btnFullscreen");
  const btnOverview = document.getElementById("btnOverview");
  const overviewModal = document.getElementById("overview-modal");
  const overviewGrid = document.getElementById("overviewGrid");
  const btnCloseOverview = document.getElementById("btnCloseOverview");

  // Populate overview modal with slide cards
  function buildOverviewGrid() {
    if (!overviewGrid) return;
    overviewGrid.innerHTML = "";
    slides.forEach((slide, idx) => {
      const card = document.createElement("div");
      card.className = `thumb-card ${idx === currentIndex ? "active-thumb" : ""}`;
      const title = slide.getAttribute("data-title") || `Slide ${idx + 1}`;
      card.innerHTML = `
        <div class="thumb-num">${String(idx + 1).padStart(2, "0")} / ${String(totalSlides).padStart(2, "0")}</div>
        <div class="thumb-title">${title}</div>
      `;
      card.addEventListener("click", () => {
        goToSlide(idx);
        toggleOverview(false);
      });
      overviewGrid.appendChild(card);
    });
  }

  function toggleOverview(forceState) {
    if (!overviewModal) return;
    const shouldOpen = typeof forceState === "boolean" 
      ? forceState 
      : !overviewModal.classList.contains("open");
    
    if (shouldOpen) {
      buildOverviewGrid();
      overviewModal.classList.add("open");
    } else {
      overviewModal.classList.remove("open");
    }
  }

  function updateUI() {
    // Hide all slides, show active slide
    slides.forEach((slide, idx) => {
      if (idx === currentIndex) {
        slide.classList.add("active");
      } else {
        slide.classList.remove("active");
      }
    });

    // Update Counter (e.g. 03 / 12)
    if (counterEl) {
      const cur = String(currentIndex + 1).padStart(2, "0");
      const tot = String(totalSlides).padStart(2, "0");
      counterEl.textContent = `${cur} / ${tot}`;
    }

    // Update Progress Bar
    if (progressBar) {
      const percent = ((currentIndex + 1) / totalSlides) * 100;
      progressBar.style.width = `${percent}%`;
    }

    // Sync URL hash
    window.location.hash = `#${currentIndex + 1}`;

    // Update overview thumbnail highlighting if open
    const thumbs = document.querySelectorAll(".thumb-card");
    thumbs.forEach((t, i) => {
      if (i === currentIndex) {
        t.classList.add("active-thumb");
      } else {
        t.classList.remove("active-thumb");
      }
    });
  }

  function goToSlide(index) {
    if (index < 0) index = 0;
    if (index >= totalSlides) index = totalSlides - 1;
    currentIndex = index;
    updateUI();
  }

  function nextSlide() {
    if (currentIndex < totalSlides - 1) {
      goToSlide(currentIndex + 1);
    }
  }

  function prevSlide() {
    if (currentIndex > 0) {
      goToSlide(currentIndex - 1);
    }
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch((err) => {
        console.warn("Fullscreen request error:", err);
      });
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }

  // Keyboard Event Listener
  window.addEventListener("keydown", (e) => {
    // If overview modal is open, Esc closes it
    if (overviewModal && overviewModal.classList.contains("open")) {
      if (e.key === "Escape" || e.key === "o" || e.key === "O") {
        e.preventDefault();
        toggleOverview(false);
        return;
      }
    }

    switch (e.key) {
      case "ArrowRight":
      case " ":
      case "PageDown":
        e.preventDefault();
        nextSlide();
        break;
      case "ArrowLeft":
      case "PageUp":
        e.preventDefault();
        prevSlide();
        break;
      case "Home":
        e.preventDefault();
        goToSlide(0);
        break;
      case "End":
        e.preventDefault();
        goToSlide(totalSlides - 1);
        break;
      case "f":
      case "F":
        e.preventDefault();
        toggleFullscreen();
        break;
      case "o":
      case "O":
        e.preventDefault();
        toggleOverview();
        break;
      case "Escape":
        if (overviewModal && overviewModal.classList.contains("open")) {
          toggleOverview(false);
        }
        break;
      default:
        break;
    }
  });

  // Button clicks
  if (btnPrev) btnPrev.addEventListener("click", prevSlide);
  if (btnNext) btnNext.addEventListener("click", nextSlide);
  if (btnFullscreen) btnFullscreen.addEventListener("click", toggleFullscreen);
  if (btnOverview) btnOverview.addEventListener("click", () => toggleOverview());
  if (btnCloseOverview) btnCloseOverview.addEventListener("click", () => toggleOverview(false));

  // Initialize from URL hash if valid
  const initialHash = window.location.hash.replace("#", "");
  const parsedSlideNum = parseInt(initialHash, 10);
  if (!isNaN(parsedSlideNum) && parsedSlideNum >= 1 && parsedSlideNum <= totalSlides) {
    currentIndex = parsedSlideNum - 1;
  } else {
    currentIndex = 0;
  }

  // Initial draw
  updateUI();
})();

