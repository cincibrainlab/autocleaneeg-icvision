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
  }

  disconnect() {
    document.removeEventListener("keydown", this.boundKeyHandler)
  }

  handleKeydown(event) {
    // Ignore if user is typing in an input
    if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") {
      return
    }

    const key = event.key.toLowerCase()

    // Number keys 1-7 for labels
    if (key >= "1" && key <= "7") {
      event.preventDefault()
      const index = parseInt(key) - 1
      this.submitLabel(LABELS[index])
      return
    }

    // Letter shortcuts
    if (LETTER_MAP[key]) {
      event.preventDefault()
      this.submitLabel(LETTER_MAP[key])
      return
    }

    // Special keys
    switch (key) {
      case "?":
        event.preventDefault()
        this.flag()
        break
      case "arrowleft":
      case "backspace":
        event.preventDefault()
        this.undo()
        break
    }
  }

  submit(event) {
    const label = event.params.label
    this.submitLabel(label)
  }

  submitLabel(label) {
    const form = document.getElementById("rating-form")
    const labelInput = document.getElementById("rating-label")
    const timeInput = document.getElementById("response-time")

    labelInput.value = label
    timeInput.value = Date.now() - this.startTimeValue

    form.requestSubmit()
  }

  flag() {
    // For now, submit with flagged=true and accept the model's label if available
    const form = document.getElementById("rating-form")
    const labelInput = document.getElementById("rating-label")
    const flagInput = document.getElementById("rating-flagged")
    const timeInput = document.getElementById("response-time")

    // Use "other_artifact" as default when flagging without a clear label
    labelInput.value = "other_artifact"
    flagInput.value = "true"
    timeInput.value = Date.now() - this.startTimeValue

    form.requestSubmit()
  }

  undo() {
    // Submit a DELETE request to undo the last rating
    fetch("/ratings/0", {
      method: "DELETE",
      headers: {
        "X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content,
        "Accept": "text/vnd.turbo-stream.html"
      }
    }).then(() => {
      window.location.reload()
    })
  }
}
