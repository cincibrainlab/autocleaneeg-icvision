class CreateRatings < ActiveRecord::Migration[8.1]
  def change
    create_table :ratings do |t|
      t.references :participant, null: false, foreign_key: true
      t.references :component, null: false, foreign_key: true
      t.string :human_label
      t.integer :response_time_ms
      t.boolean :flagged
      t.string :session_id

      t.timestamps
    end
  end
end
