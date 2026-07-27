import AppKit
import CoreGraphics
import Foundation

guard CommandLine.arguments.count == 3 else {
    fputs("Usage: render_markdown_pdf input.md output.pdf\n", stderr)
    exit(2)
}

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
let source = try String(contentsOf: inputURL, encoding: .utf8)

let pageWidth: CGFloat = 612
let pageHeight: CGFloat = 792
let margin: CGFloat = 54
let contentWidth = pageWidth - 2 * margin
let colorSpace = CGColorSpaceCreateDeviceRGB()
guard let consumer = CGDataConsumer(url: outputURL as CFURL),
      let context = CGContext(consumer: consumer, mediaBox: nil, nil) else {
    fputs("Unable to create PDF context\n", stderr)
    exit(1)
}

let bodyFont = NSFont.systemFont(ofSize: 10.5)
let heading1Font = NSFont.boldSystemFont(ofSize: 23)
let heading2Font = NSFont.boldSystemFont(ofSize: 15)
let codeFont = NSFont.monospacedSystemFont(ofSize: 9.5, weight: .regular)
let bulletFont = NSFont.systemFont(ofSize: 10.5)
let textColor = NSColor(calibratedWhite: 0.12, alpha: 1)
let accentColor = NSColor(calibratedRed: 0.08, green: 0.27, blue: 0.48, alpha: 1)
let codeBackground = NSColor(calibratedWhite: 0.95, alpha: 1)

var y: CGFloat = pageHeight - margin
var pageNumber = 0
var inCode = false

func startPage() {
    var mediaBox = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)
    context.beginPDFPage([kCGPDFContextMediaBox as String: NSData(bytes: &mediaBox, length: MemoryLayout<CGRect>.size)] as CFDictionary)
    context.setFillColor(NSColor.white.cgColor)
    context.fill(mediaBox)
    y = pageHeight - margin
    pageNumber += 1
}

func finishPage() {
    let footer = "NDI Robot Base Calibration   |   Page \(pageNumber)"
    let attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 8),
        .foregroundColor: NSColor.gray
    ]
    let footerString = NSAttributedString(string: footer, attributes: attrs)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
    footerString.draw(at: CGPoint(x: margin, y: 25))
    NSGraphicsContext.restoreGraphicsState()
    context.endPDFPage()
}

func paragraphStyle(spacing: CGFloat, indent: CGFloat = 0) -> NSMutableParagraphStyle {
    let style = NSMutableParagraphStyle()
    style.lineSpacing = spacing
    style.firstLineHeadIndent = indent
    style.headIndent = indent
    return style
}

func drawBlock(_ text: String, font: NSFont, color: NSColor, before: CGFloat, after: CGFloat, indent: CGFloat = 0, background: NSColor? = nil) {
    let availableWidth = contentWidth - indent
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: color,
        .paragraphStyle: paragraphStyle(spacing: font == codeFont ? 2 : 2.5)
    ]
    let attributed = NSAttributedString(string: text, attributes: attrs)
    let bounds = attributed.boundingRect(
        with: NSSize(width: availableWidth - (background == nil ? 0 : 16), height: 10_000),
        options: [.usesLineFragmentOrigin, .usesFontLeading]
    )
    let blockHeight = ceil(bounds.height) + (background == nil ? 0 : 14)
    if y - before - blockHeight - after < margin + 16 {
        finishPage()
        startPage()
    }
    y -= before + blockHeight
    let rect = CGRect(x: margin + indent, y: y, width: availableWidth, height: blockHeight)
    if let background {
        context.setFillColor(background.cgColor)
        context.fill(rect.insetBy(dx: 0, dy: 0))
    }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
    let textRect = background == nil ? rect : rect.insetBy(dx: 8, dy: 7)
    attributed.draw(with: textRect, options: [.usesLineFragmentOrigin, .usesFontLeading])
    NSGraphicsContext.restoreGraphicsState()
    y -= after
}

startPage()

for rawLine in source.components(separatedBy: .newlines) {
    let line = rawLine.trimmingCharacters(in: .whitespaces)
    if line == "```text" || line == "```" {
        inCode.toggle()
        continue
    }
    if line.isEmpty {
        y -= inCode ? 3 : 5
        continue
    }
    if inCode {
        drawBlock(rawLine, font: codeFont, color: textColor, before: 0, after: 0, indent: 0, background: codeBackground)
    } else if line.hasPrefix("# ") {
        drawBlock(String(line.dropFirst(2)), font: heading1Font, color: accentColor, before: 0, after: 10)
    } else if line.hasPrefix("## ") {
        drawBlock(String(line.dropFirst(3)), font: heading2Font, color: accentColor, before: 10, after: 5)
    } else if line.hasPrefix("- ") {
        drawBlock("• " + String(line.dropFirst(2)), font: bulletFont, color: textColor, before: 1, after: 1, indent: 12)
    } else {
        drawBlock(line, font: bodyFont, color: textColor, before: 1, after: 2)
    }
}

finishPage()
context.closePDF()
print("Created \(outputURL.path)")
