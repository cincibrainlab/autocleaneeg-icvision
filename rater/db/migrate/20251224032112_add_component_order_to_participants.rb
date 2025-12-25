class AddComponentOrderToParticipants < ActiveRecord::Migration[8.1]
  def change
    add_column :participants, :component_order, :text
  end
end
