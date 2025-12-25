import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = ["dropdown"]
  static values = { url: String }

  connect() {
    this.boundClickOutside = this.clickOutside.bind(this)
    document.addEventListener("click", this.boundClickOutside)
  }

  disconnect() {
    document.removeEventListener("click", this.boundClickOutside)
  }

  toggle(event) {
    event.stopPropagation()
    this.dropdownTarget.classList.toggle("hidden")
  }

  clickOutside(event) {
    if (!this.element.contains(event.target)) {
      this.dropdownTarget.classList.add("hidden")
    }
  }

  copy(event) {
    const url = event.currentTarget.dataset.resumeLinkUrlValue
    navigator.clipboard.writeText(url).then(() => {
      const button = event.currentTarget
      const originalText = button.textContent
      button.textContent = "Copied!"
      button.classList.remove("bg-blue-600", "hover:bg-blue-500")
      button.classList.add("bg-green-600")

      setTimeout(() => {
        button.textContent = originalText
        button.classList.remove("bg-green-600")
        button.classList.add("bg-blue-600", "hover:bg-blue-500")
      }, 2000)
    })
  }
}
