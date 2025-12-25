Rails.application.routes.draw do
  # Main rating flow
  root "ratings#index"

  # Token-based resume link
  get "r/:token", to: "ratings#resume", as: :resume_session

  # Rating endpoints
  post "ratings", to: "ratings#create"
  get "rate", to: "ratings#show"  # Shows current component to rate
  patch "ratings/:id/flag", to: "ratings#flag", as: :flag_rating
  delete "ratings/:id", to: "ratings#undo", as: :undo_rating
  post "ratings/undo", to: "ratings#undo"  # POST endpoint for JS undo

  # Participant management
  patch "participant/experience", to: "participants#update_experience", as: :update_experience

  # Progress/completion
  get "complete", to: "ratings#complete"
  get "progress", to: "ratings#progress"

  # Health check
  get "up" => "rails/health#show", as: :rails_health_check
end
