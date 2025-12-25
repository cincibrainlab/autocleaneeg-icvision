// Configure your import map in config/importmap.rb. Read more: https://github.com/rails/importmap-rails
import "@hotwired/turbo-rails"
import { Turbo } from "@hotwired/turbo-rails"
import "controllers"

// Configure Turbo for smooth navigation
Turbo.setProgressBarDelay(0)

// Ensure scroll position is always at top for rating pages
document.addEventListener("turbo:load", () => {
  window.scrollTo(0, 0)
})
