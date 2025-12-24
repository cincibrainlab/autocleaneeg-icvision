module ApplicationHelper
  def experience_color(level)
    case level
    when "novice"
      "bg-green-400"
    when "trained"
      "bg-blue-400"
    when "expert"
      "bg-purple-400"
    else
      "bg-gray-400"
    end
  end
end
