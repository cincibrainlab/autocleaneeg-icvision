class Participant < ApplicationRecord
  has_many :ratings, dependent: :destroy

  EXPERIENCE_LEVELS = %w[novice trained expert].freeze

  validates :uuid, presence: true, uniqueness: true
  validates :token, presence: true, uniqueness: true
  validates :experience_level, inclusion: { in: EXPERIENCE_LEVELS }, allow_nil: true

  before_validation :generate_uuid, on: :create
  before_validation :generate_token, on: :create
  after_create :generate_component_order

  serialize :component_order, coder: JSON

  def experience_set?
    experience_level.present?
  end

  # Short URL-friendly token for sharing
  def short_token
    token[0..7]
  end

  # Get the next component to rate based on this participant's randomized order
  def next_component
    return nil if component_order.blank?

    rated_ids = ratings.pluck(:component_id)
    remaining_ids = component_order - rated_ids
    return nil if remaining_ids.empty?

    Component.find_by(id: remaining_ids.first)
  end

  # Regenerate component order (useful if new components are added)
  def refresh_component_order!
    generate_component_order
    save!
  end

  private

  def generate_uuid
    self.uuid ||= SecureRandom.uuid
  end

  def generate_token
    self.token ||= SecureRandom.urlsafe_base64(16)
  end

  def generate_component_order
    self.component_order = Component.pluck(:id).shuffle
  end
end
