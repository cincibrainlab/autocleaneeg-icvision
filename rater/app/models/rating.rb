class Rating < ApplicationRecord
  belongs_to :participant
  belongs_to :component

  LABELS = Component::LABELS

  validates :human_label, inclusion: { in: LABELS }
  validates :response_time_ms, numericality: { greater_than: 0 }, allow_nil: true
  validates :session_id, presence: true
  validates :component_id, uniqueness: { scope: :participant_id, message: "already rated by this participant" }

  scope :flagged, -> { where(flagged: true) }
  scope :for_session, ->(session_id) { where(session_id: session_id) }
end
