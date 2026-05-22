/* eralyzer — static reference data
   DEMOS:     dropdown options only — no scripts (API provides live scripts)
   ERAS:      era metadata for colours + confusion matrix labels
   CONFUSION: placeholder matrix (replaced by real model data when trained)
*/

const ERAS = [
  { id: "debut",     label: "Debut",      year: 2006, color: "#c8a97e" },
  { id: "fearless",  label: "Fearless",   year: 2008, color: "#d9a818" },
  { id: "speaknow",  label: "Speak Now",  year: 2010, color: "#9a6cc1" },
  { id: "red",       label: "Red",        year: 2012, color: "#b53030" },
  { id: "1989",      label: "1989",       year: 2014, color: "#4a9fc7" },
  { id: "rep",       label: "Reputation", year: 2017, color: "#4a4a4a" },
  { id: "lover",     label: "Lover",      year: 2019, color: "#db83b0" },
  { id: "folklore",  label: "Folklore",   year: 2020, color: "#6f8a5c" },
  { id: "evermore",  label: "Evermore",   year: 2020, color: "#b56830" },
  { id: "midnights", label: "Midnights",  year: 2022, color: "#4a5aa8" },
];

/* Demo songs — one per era, all confirmed present in the dataset (224 songs, all 10 eras). */
const DEMOS = [
  { era: "debut",     title: "Tim McGraw",          artist: "Taylor Swift" },
  { era: "fearless",  title: "Love Story",          artist: "Taylor Swift" },
  { era: "speaknow",  title: "Enchanted",           artist: "Taylor Swift" },
  { era: "red",       title: "All Too Well",        artist: "Taylor Swift" },
  { era: "1989",      title: "Blank Space",         artist: "Taylor Swift" },
  { era: "rep",       title: "Getaway Car",         artist: "Taylor Swift" },
  { era: "lover",     title: "Cruel Summer",        artist: "Taylor Swift" },
  { era: "folklore",  title: "cardigan",            artist: "Taylor Swift" },
  { era: "evermore",  title: "champagne problems",  artist: "Taylor Swift" },
  { era: "midnights", title: "Lavender Haze",       artist: "Taylor Swift" },
];

/* Confusion matrix: rows = true era, cols = predicted era (fractions 0..1).
   Hand-tuned placeholder — replaced by real model output after training. */
const CONFUSION = [
  /*           debut  fear  speak  red   1989  rep   lover folk  ever  mid  */
  /* debut  */ [0.74, 0.16, 0.04, 0.03, 0.01, 0.00, 0.00, 0.01, 0.01, 0.00],
  /* fear   */ [0.11, 0.62, 0.18, 0.06, 0.01, 0.00, 0.00, 0.01, 0.01, 0.00],
  /* speak  */ [0.03, 0.19, 0.58, 0.13, 0.03, 0.01, 0.01, 0.01, 0.01, 0.00],
  /* red    */ [0.02, 0.06, 0.11, 0.55, 0.16, 0.04, 0.03, 0.01, 0.01, 0.01],
  /* 1989   */ [0.00, 0.02, 0.03, 0.13, 0.61, 0.10, 0.08, 0.01, 0.01, 0.01],
  /* rep    */ [0.00, 0.00, 0.01, 0.04, 0.11, 0.59, 0.18, 0.01, 0.01, 0.05],
  /* lover  */ [0.00, 0.01, 0.01, 0.02, 0.09, 0.13, 0.64, 0.02, 0.02, 0.06],
  /* folk   */ [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.49, 0.38, 0.03],
  /* ever   */ [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.42, 0.46, 0.02],
  /* mid    */ [0.00, 0.00, 0.01, 0.02, 0.04, 0.07, 0.09, 0.02, 0.02, 0.73],
];
