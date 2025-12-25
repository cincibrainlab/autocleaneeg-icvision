import { Controller } from "@hotwired/stimulus"

// Label mapping for keyboard shortcuts
const LABELS = ["brain", "eye", "muscle", "heart", "line_noise", "channel_noise", "other_artifact"]
const LETTER_MAP = {
  "b": "brain",
  "e": "eye",
  "m": "muscle",
  "h": "heart",
  "l": "line_noise",
  "c": "channel_noise",
  "o": "other_artifact"
}

export default class extends Controller {
  static values = {
    componentId: Number,
    startTime: Number
  }

  connect() {
    this.boundKeyHandler = this.handleKeydown.bind(this)
    document.addEventListener("keydown", this.boundKeyHandler)
    this.isSubmitting = false
    this.isUndoing = false
  }

  disconnect() {
    document.removeEventListener("keydown", this.boundKeyHandler)
  }

  handleKeydown(event) {
    // Ignore if user is typing in an input
    if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") {
      return
    }

    // Prevent actions while submitting or undoing
    if (this.isSubmitting || this.isUndoing) {
      event.preventDefault()
      return
    }

    // Normalize key to lowercase for consistent handling
    const key = event.key.toLowerCase()

    // Number keys 1-7 for labels
    if (key >= "1" && key <= "7") {
      event.preventDefault()
      const index = parseInt(key) - 1
      this.submitLabel(LABELS[index])
      return
    }

    // Letter shortcuts (case-insensitive due to toLowerCase above)
    if (LETTER_MAP[key]) {
      event.preventDefault()
      this.submitLabel(LETTER_MAP[key])
      return
    }

    // Flag shortcut: 'f' key OR '?' (Shift+/)
    if (key === "f" || event.key === "?") {
      event.preventDefault()
      this.flag()
      return
    }

    // Undo shortcuts: ArrowLeft or Backspace
    if (key === "arrowleft" || key === "backspace") {
      event.preventDefault()
      this.undo()
      return
    }
  }

  submit(event) {
    if (this.isSubmitting) return
    const label = event.params.label
    this.submitLabel(label)
  }

  submitLabel(label) {
    // Prevent double submissions
    if (this.isSubmitting) return
    this.isSubmitting = true

    const form = document.getElementById("rating-form")
    const labelInput = document.getElementById("rating-label")
    const timeInput = document.getElementById("response-time")

    labelInput.value = label
    timeInput.value = Date.now() - this.startTimeValue

    // Disable buttons during submission
    this.disableButtons()

    form.requestSubmit()
  }

  flag() {
    // Prevent double submissions
    if (this.isSubmitting) return
    this.isSubmitting = true

    const form = document.getElementById("rating-form")
    const labelInput = document.getElementById("rating-label")
    const flagInput = document.getElementById("rating-flagged")
    const timeInput = document.getElementById("response-time")

    // Use "other_artifact" as default when flagging without a clear label
    labelInput.value = "other_artifact"
    flagInput.value = "true"
    timeInput.value = Date.now() - this.startTimeValue

    // Disable buttons during submission
    this.disableButtons()

    form.requestSubmit()
  }

  undo() {
    // Prevent multiple undo requests
    if (this.isUndoing) return
    this.isUndoing = true

    // Visual feedback - disable undo button
    const undoBtn = document.querySelector('[data-action*="rating#undo"]')
    if (undoBtn) {
      undoBtn.disabled = true
      undoBtn.classList.add("opacity-50", "cursor-not-allowed")
    }

    fetch("/ratings/undo", {
      method: "POST",
      headers: {
        "X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content,
        "Accept": "text/html"
      }
    }).then((response) => {
      if (response.ok) {
        window.location.reload()
      } else {
        // Re-enable on error
        this.isUndoing = false
        if (undoBtn) {
          undoBtn.disabled = false
          undoBtn.classList.remove("opacity-50", "cursor-not-allowed")
        }
      }
    }).catch(() => {
      this.isUndoing = false
      if (undoBtn) {
        undoBtn.disabled = false
        undoBtn.classList.remove("opacity-50", "cursor-not-allowed")
      }
    })
  }

  disableButtons() {
    const buttons = document.querySelectorAll('.rating-btn, [data-action*="rating#"]')
    buttons.forEach(btn => {
      btn.disabled = true
      btn.classList.add("opacity-50", "cursor-not-allowed")
    })
  }
}
