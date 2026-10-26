// pip_track.py 调用：用 macOS 自带的 Vision 找每张图里最大的那张脸，打印一行一张图：
//   文件名,x,y,w,h（0—1 的比例，原点在左上角）；找不到脸就只打印文件名。
// 用法：swift scripts/face_track.swift <放 jpg 的文件夹>
import Foundation
import Vision
import ImageIO

let dir = URL(fileURLWithPath: CommandLine.arguments[1])
let files = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".jpg") }.sorted()
for name in files {
    let url = dir.appendingPathComponent(name)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { print(name); continue }
    let req = VNDetectFaceRectanglesRequest()
    try? VNImageRequestHandler(cgImage: img, options: [:]).perform([req])
    let faces = (req.results ?? []).sorted { $0.boundingBox.width * $0.boundingBox.height > $1.boundingBox.width * $1.boundingBox.height }
    if let f = faces.first {
        let b = f.boundingBox  // Vision 的原点在左下角
        print(String(format: "%@,%.4f,%.4f,%.4f,%.4f", name, b.minX, 1 - b.maxY, b.width, b.height))
    } else {
        print(name)
    }
}
