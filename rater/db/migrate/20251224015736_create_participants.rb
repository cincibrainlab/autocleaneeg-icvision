class CreateParticipants < ActiveRecord::Migration[8.1]
  def change
    create_table :participants do |t|
      t.string :uuid
      t.string :experience_level

      t.timestamps
    end
    add_index :participants, :uuid
  end
end
