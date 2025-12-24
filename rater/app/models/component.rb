class Component < ApplicationRecord
  has_many :ratings, dependent: :destroy

  LABELS = %w[brain eye muscle heart line_noise channel_noise other_artifact].freeze

  validates :ic_index, presence: true
  validates :image_path, presence: true
  validates :model_label, inclusion: { in: LABELS }, allow_nil: true
  validates :model_confidence, numericality: { greater_than_or_equal_to: 0, less_than_or_equal_to: 1 }, allow_nil: true

  scope :for_dataset, ->(dataset) { where(dataset: dataset) }
  scope :unrated_by, ->(participant) {
    left_joins(:ratings)
      .where(ratings: { id: nil })
      .or(left_joins(:ratings).where.not(ratings: { participant_id: participant.id }))
      .distinct
  }
end
