# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# This file is the source Rails uses to define your schema when running `bin/rails
# db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
# be faster and is potentially less error prone than running all of your
# migrations from scratch. Old migrations may fail to apply correctly if those
# migrations use external dependencies or application code.
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema[8.1].define(version: 2025_12_24_032112) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "pg_catalog.plpgsql"

  create_table "components", force: :cascade do |t|
    t.datetime "created_at", null: false
    t.string "dataset"
    t.integer "ic_index"
    t.string "image_path"
    t.float "model_confidence"
    t.string "model_label"
    t.datetime "updated_at", null: false
  end

  create_table "participants", force: :cascade do |t|
    t.text "component_order"
    t.datetime "created_at", null: false
    t.string "experience_level"
    t.string "token"
    t.datetime "updated_at", null: false
    t.string "uuid"
    t.index ["token"], name: "index_participants_on_token"
    t.index ["uuid"], name: "index_participants_on_uuid"
  end

  create_table "ratings", force: :cascade do |t|
    t.bigint "component_id", null: false
    t.datetime "created_at", null: false
    t.boolean "flagged"
    t.string "human_label"
    t.bigint "participant_id", null: false
    t.integer "response_time_ms"
    t.string "session_id"
    t.datetime "updated_at", null: false
    t.index ["component_id"], name: "index_ratings_on_component_id"
    t.index ["participant_id"], name: "index_ratings_on_participant_id"
  end

  add_foreign_key "ratings", "components"
  add_foreign_key "ratings", "participants"
end
