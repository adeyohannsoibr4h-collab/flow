// netlify/functions/ask.js
// Fonction serveur de FlowAI : reçoit la question de l'élève et interroge Groq (API compatible OpenAI).
// La clé API reste côté serveur, dans les variables d'environnement Netlify (jamais dans index.html).

const API_URL = "https://api.groq.com/openai/v1/chat/completions";
const MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
// Modèle utilisé quand l'élève joint une image (il doit accepter les images)
const VISION_MODEL = process.env.GROQ_VISION_MODEL || "meta-llama/llama-4-scout-17b-16e-instruct";

const json = (statusCode, body) => ({
  statusCode,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body)
});

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return json(405, { error: "Méthode non autorisée" });
  }

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return json(500, { error: "GROQ_API_KEY manquante sur le serveur" });
  }

  let data;
  try {
    data = JSON.parse(event.body || "{}");
  } catch {
    return json(400, { error: "Requête invalide" });
  }

  const question  = String(data.question  || "").trim().slice(0, 9000);
  const niveau    = String(data.niveau    || "non précisé").slice(0, 40);
  const programme = String(data.programme || "non précisé").slice(0, 40);
  const matiere   = String(data.matiere   || "Toutes matières").slice(0, 40);

  if (!question) {
    return json(400, { error: "Question vide" });
  }

  // Images jointes (max 4, format data:image/...;base64, ~3,6 Mo au total)
  const images = (Array.isArray(data.images) ? data.images : [])
    .filter(i => typeof i === "string" && i.startsWith("data:image/"))
    .slice(0, 4);
  if (images.reduce((n, i) => n + i.length, 0) > 3600000) {
    return json(413, { error: "Images trop lourdes" });
  }

  // Historique récent envoyé par le site (rôle "ai" -> "assistant")
  const historique = Array.isArray(data.historique) ? data.historique.slice(-6) : [];
  const previous = historique
    .filter(m => m && typeof m.content === "string")
    .map(m => ({
      role: m.role === "user" ? "user" : "assistant",
      content: m.content.slice(0, 2000)
    }));

  const system = `Tu es FlowAI, un professeur particulier bienveillant qui aide des élèves.
Contexte de l'élève : classe de ${niveau}, ${programme}, matière : ${matiere}.

Règles :
- Réponds toujours en français, avec des mots simples adaptés au niveau ${niveau}.
- Explique pas à pas, avec un exemple concret quand c'est utile (de préférence proche de la vie au Gabon si le programme est gabonais).
- Pour un exercice, guide l'élève vers la solution : donne la méthode, puis la correction détaillée si l'élève la demande ou semble bloqué.
- Si une image est jointe (photo d'un exercice, d'un cahier ou d'un schéma), lis-la attentivement, recopie l'énoncé important, puis aide l'élève. Si elle est floue ou illisible, demande une meilleure photo.
- Reste dans le cadre scolaire. Si la question n'a aucun rapport avec les études, ramène poliment la conversation vers l'école.
- Si tu n'es pas sûr d'une information, dis-le au lieu d'inventer.
- Écris en texte simple, sans tableaux ni syntaxe Markdown compliquée (pas de ** ni de #). Utilise des tirets et des retours à la ligne.
- Termine par une courte question ou une mini-vérification pour s'assurer que l'élève a compris.`;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: images.length ? VISION_MODEL : MODEL,
        temperature: 0.4,
        max_tokens: 900,
        messages: [
          { role: "system", content: system },
          ...previous,
          {
            role: "user",
            content: images.length
              ? [{ type: "text", text: question }, ...images.map(url => ({ type: "image_url", image_url: { url } }))]
              : question
          }
        ]
      })
    });

    const result = await response.json();

    if (!response.ok) {
      console.error("Erreur Groq :", response.status, JSON.stringify(result));
      const message = response.status === 429
        ? "Trop de questions en même temps. Réessaie dans une minute."
        : "Le service IA est indisponible pour le moment.";
      return json(response.status === 429 ? 429 : 502, { error: message });
    }

    // On renvoie le format attendu par index.html : data.choices[0].message.content
    return json(200, result);
  } catch (err) {
    console.error(err);
    return json(500, { error: "Erreur interne du serveur" });
  }
};