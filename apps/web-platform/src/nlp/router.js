// Turns an intent and free text into structured parameters, a severity and device actions.
const UNITS = { c: "celsius", "°c": "celsius", celsius: "celsius", degree: "celsius", degrees: "celsius", "%": "percent", lux: "lux", v: "volt", volt: "volt", volts: "volt" };
const TARGETS = [[/\b(fan|cooling|ventilation)\b/i, "fan"], [/\b(led|lamp|light|lights|lighting|illumination)\b/i, "led"], [/\b(relay)\b/i, "relay"], [/\b(pump)\b/i, "pump"], [/\b(heater)\b/i, "heater"], [/\b(valve)\b/i, "valve"], [/\b(motor)\b/i, "motor"]];
const SENSORS = [[/\b(temperature|temp|hot|warm|heat|overheating)\b/i, "temperature"], [/\b(light|illumination|lux|dark|dim|brightness)\b/i, "light"], [/\b(vibration)\b/i, "vibration"], [/\b(humidity)\b/i, "humidity"]];
export function extract(text) {
  const measurements = [...text.matchAll(/(-?\d+(?:\.\d+)?)\s*(°\s?c|celsius|degrees?|%|lux|volts?|v|c)\b/gi)].map((m) => ({ value: +m[1], unit: UNITS[m[2].toLowerCase().replace(/\s/g, "")] || m[2] }));
  const dur = text.match(/(\d+|one|two|three|five|ten|fifteen|thirty)\s*(seconds?|minutes?|hours?)/i);
  const nums = { one: 1, two: 2, three: 3, five: 5, ten: 10, fifteen: 15, thirty: 30 };
  return {
    sensor: (SENSORS.find(([r]) => r.test(text)) || [])[1] || null,
    trend: /\b(ris(e|ing)|climb\w*|increas\w*|exceed\w*|above|spik\w*|too hot|warm)\b/i.test(text) ? "rising" : /\b(drop\w*|fall\w*|fell|decreas\w*|below|dim|dark\w*|under)\b/i.test(text) ? "falling" : null,
    heatHigh: /\b(temperature|temp|heat|hot|warm|overheat\w*)\b[^,.]*\b(ris\w*|climb\w*|exceed\w*|high|hot\w*|spik\w*|overheat\w*)\b/i.test(text),
    lightLow: /\b(illumination|light\w*|lux|brightness)\b[^,.]*\b(drop\w*|fell|fall\w*|low|dim|dark\w*)\b/i.test(text),
    rapid: /\b(fast|rapid\w*|quick\w*|sudden\w*|sharp\w*)\b/i.test(text),
    targets: TARGETS.filter(([r]) => r.test(text)).map(([, n]) => n),
    state: /\b(turn on|switch on|activate|start|enable|open|trigger)\b/i.test(text) ? "on" : /\b(turn off|switch off|shut|stop|disable|close)\b/i.test(text) ? "off" : null,
    location: (text.match(/\b(chamber|room|tank|greenhouse|rack|lab|oven)\b/i) || [])[1]?.toLowerCase() || null,
    measurements, duration: dur ? { value: nums[dur[1].toLowerCase()] ?? +dur[1], unit: dur[2].toLowerCase().replace(/s$/, "") } : null,
  };
}
export function severity(intent, text, p) {
  let s = { anomaly_report: 2, temperature_alert: 1, lighting_alert: 1, actuator_control: 0, status_query: 0, maintenance: 0 }[intent] ?? 0;
  if (/\b(critical|urgent|immediately|emergency|overheat\w*|fire|danger\w*)\b/i.test(text)) s += 2;
  if (p.rapid) s += 1; if (p.trend) s += 1;
  const t = p.measurements.find((m) => m.unit === "celsius"); if (t && t.value >= 40) s += 2;
  return s >= 4 ? "high" : s >= 2 ? "medium" : "low";
}
export function plan(intent, p, sev) {
  const a = [];
  if (intent === "temperature_alert" && p.trend !== "falling") a.push({ target: "fan", state: "on" });
  if (intent === "lighting_alert" && p.trend === "falling") a.push({ target: "led", state: "on" });
  if (intent === "lighting_alert" && p.trend === "rising") a.push({ target: "led", state: "off" });
  if (intent === "actuator_control") (p.targets.length ? p.targets : ["relay"]).forEach((t) => a.push({ target: t, state: p.state || "on", ...(p.duration ? { duration: p.duration } : {}) }));
  if (["temperature_alert", "lighting_alert", "anomaly_report"].includes(intent)) { // secondary conditions named in the same note
    if (p.heatHigh) a.push({ target: "fan", state: "on" });
    if (p.lightLow) a.push({ target: "led", state: "on" });
  }
  if (intent === "anomaly_report" || sev !== "low") a.push({ target: "email", state: "notify" });
  if (intent === "status_query") a.push({ target: "telemetry", state: "report" });
  if (intent === "maintenance") a.push({ target: "ticket", state: "create" });
  return a.filter((x, i, arr) => arr.findIndex((y) => y.target === x.target && y.state === x.state) === i);
}
