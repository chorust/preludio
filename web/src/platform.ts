type UserAgentDataLike = {
  brands?: readonly { brand: string; version?: string }[];
};

type NavigatorWithUserAgentData = Navigator & {
  userAgentData?: UserAgentDataLike;
};

export function detectPlatform(): string {
  if (typeof navigator === "undefined") return "web-unknown";

  const nav = navigator as NavigatorWithUserAgentData;
  const brands = nav.userAgentData?.brands?.map(({ brand }) => brand.toLowerCase()) ?? [];
  if (brands.some((brand) => brand.includes("chrome") || brand.includes("chromium"))) {
    return "web-chrome";
  }

  const userAgent = nav.userAgent ?? "";
  if (/Firefox\//i.test(userAgent) || /FxiOS\//i.test(userAgent)) return "web-firefox";
  if (/Chrome\//i.test(userAgent) || /Chromium\//i.test(userAgent) || /CriOS\//i.test(userAgent)) {
    return "web-chrome";
  }
  if (/Safari\//i.test(userAgent)) return "web-safari";
  return "web-unknown";
}

export function isCrossOriginIsolated(): boolean {
  return Boolean((globalThis as typeof globalThis & { crossOriginIsolated?: boolean }).crossOriginIsolated);
}
