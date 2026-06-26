import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://dorc.vercel.app"),
  title: "DORC | Image Classifier",
  description: "An image classifier model based on EfficientNet-B0 — 108 classes, client-side ONNX inference.",
  openGraph: {
    title: "DORC | Image Classifier",
    description: "An image classifier model based on EfficientNet-B0 — 108 classes, client-side ONNX inference.",
    images: [
      {
        url: "/preview.png",
        width: 1280,
        height: 720,
        alt: "DORC Image Classifier Preview",
      },
    ],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "DORC | Image Classifier",
    description: "An image classifier model based on EfficientNet-B0 — 108 classes, client-side ONNX inference.",
    images: ["/preview.png"],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />
      </head>
      <body suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
