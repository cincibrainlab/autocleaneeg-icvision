class RatingsController < ApplicationController
  before_action :find_or_create_participant
  before_action :set_current_component, only: [:show, :create]

  def index
    redirect_to rate_path
  end

  def resume
    participant = Participant.find_by(token: params[:token])

    if participant
      session[:participant_uuid] = participant.uuid
      @current_participant = participant
      redirect_to rate_path, notice: "Welcome back! Resuming your session."
    else
      redirect_to root_path, alert: "Invalid session link."
    end
  end

  def show
    if @component.nil?
      redirect_to complete_path
      return
    end

    @progress = {
      completed: current_participant.ratings.count,
      total: Component.count,
      percentage: Component.count.positive? ? (current_participant.ratings.count * 100.0 / Component.count).round : 0
    }

    @resume_url = resume_session_url(current_participant.token)
  end

  def create
    @rating = current_participant.ratings.build(rating_params)
    @rating.component = @component
    @rating.session_id = session.id.to_s
    @rating.response_time_ms = params[:response_time_ms].to_i if params[:response_time_ms].present?

    if @rating.save
      respond_to do |format|
        format.turbo_stream { redirect_to rate_path }
        format.html { redirect_to rate_path }
      end
    else
      respond_to do |format|
        format.turbo_stream { render turbo_stream: turbo_stream.replace("error", partial: "error", locals: { message: @rating.errors.full_messages.join(", ") }) }
        format.html { render :show, status: :unprocessable_entity }
      end
    end
  end

  def flag
    @rating = current_participant.ratings.find(params[:id])
    @rating.update(flagged: !@rating.flagged)

    respond_to do |format|
      format.turbo_stream { redirect_to rate_path }
      format.html { redirect_to rate_path }
    end
  end

  def undo
    @rating = current_participant.ratings.order(created_at: :desc).first
    @rating&.destroy

    respond_to do |format|
      format.turbo_stream { redirect_to rate_path }
      format.html { redirect_to rate_path }
    end
  end

  def complete
    @stats = {
      total_rated: current_participant.ratings.count,
      flagged: current_participant.ratings.flagged.count,
      experience_set: current_participant.experience_set?
    }
    @resume_url = resume_session_url(current_participant.token)
  end

  def progress
    render json: {
      completed: current_participant.ratings.count,
      total: Component.count,
      percentage: Component.count.positive? ? (current_participant.ratings.count * 100.0 / Component.count).round : 0
    }
  end

  private

  def find_or_create_participant
    if session[:participant_uuid].present?
      @current_participant = Participant.find_by(uuid: session[:participant_uuid])
    end

    unless @current_participant
      @current_participant = Participant.create!
      session[:participant_uuid] = @current_participant.uuid
    end
  end

  def current_participant
    @current_participant
  end
  helper_method :current_participant

  def set_current_component
    @component = current_participant.next_component
  end

  def rating_params
    params.require(:rating).permit(:human_label, :flagged)
  end
end
