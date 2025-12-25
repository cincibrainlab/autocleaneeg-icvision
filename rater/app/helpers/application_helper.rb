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

  def qr_code_svg(url, size: 80)
    require "rqrcode"
    qr = RQRCode::QRCode.new(url)
    qr.as_svg(
      color: "fff",
      shape_rendering: "crispEdges",
      module_size: 2,
      standalone: true,
      use_path: true,
      viewbox: true,
      svg_attributes: {
        width: size,
        height: size,
        class: "inline-block"
      }
    ).html_safe
  end
end
