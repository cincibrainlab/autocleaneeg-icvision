class ParticipantsController < ApplicationController
  def update_experience
    participant = Participant.find_by(uuid: session[:participant_uuid])

    if participant&.update(experience_level: params[:experience_level])
      respond_to do |format|
        format.turbo_stream { render turbo_stream: turbo_stream.replace("experience-badge", partial: "shared/experience_badge", locals: { participant: participant }) }
        format.html { redirect_back(fallback_location: rate_path) }
      end
    else
      head :unprocessable_entity
    end
  end
end
