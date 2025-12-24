class Participant < ApplicationRecord
  has_many :ratings, dependent: :destroy

  EXPERIENCE_LEVELS = %w[novice trained expert].freeze

  validates :uuid, presence: true, uniqueness: true
  validates :experience_level, inclusion: { in: EXPERIENCE_LEVELS }, allow_nil: true

  before_validation :generate_uuid, on: :create

  def experience_set?
    experience_level.present?
  end

  private

  def generate_uuid
    self.uuid ||= SecureRandom.uuid
  end
end
