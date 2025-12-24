class CreateComponents < ActiveRecord::Migration[8.1]
  def change
    create_table :components do |t|
      t.integer :ic_index
      t.string :image_path
      t.string :model_label
      t.float :model_confidence
      t.string :dataset

      t.timestamps
    end
  end
end
