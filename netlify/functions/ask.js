// functions/api/ask.js
// FlowAI : fonction Cloudflare Pages qui transmet les questions à Groq.
// Elle répond à l'adresse /api/ask.
// Les clés se règlent dans Cloudflare (Settings > Variables and Secrets), jamais dans ce fichier :
//   GROQ_API_KEY      (obligatoire, en "Secret")
//   FIREBASE_API_KEY  (conseillé : seuls les élèves connectés peuvent poser des questions)
//   GROQ_MODEL, GROQ_VISION_MODEL (facultatifs)

const API_URL = "https://api.groq.com/openai/v1/chat/completions";

const DEFAULT_MODEL = "openai/gpt-oss-120b";
const DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b";

const MAX_QUESTION_LENGTH = 9000;
const MAX_HISTORY_MESSAGES = 6;
const MAX_HISTORY_MESSAGE_LENGTH = 2000;
const MAX_IMAGES = 3;
const MAX_IMAGES_TOTAL_CHARS = 3_500_000;
const MAX_BODY_CHARS = 5_000_000;
const MAX_OUTPUT_TOKENS = 2000;
const REQUEST_TIMEOUT_MS = 25_000;

function json(statusCode, body, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status: statusCode,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      ...extraHeaders
    }
  });
}

// Vérifie que la demande vient d'un élève connecté (jeton Firebase envoyé par le site).
async function verifyUser(request, env) {
  if (!env.FIREBASE_API_KEY) return true; // désactivé tant que la variable n'existe pas
  const header = request.headers.get("Authorization") || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : "";
  if (!token) return false;
  try {
    const r = await fetch(
      `https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=${env.FIREBASE_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ idToken: token })
      }
    );
    if (!r.ok) return false;
    const d = await r.json();
    return Array.isArray(d.users) && d.users.length > 0;
  } catch {
    return false;
  }
}

function cleanText(value, fallback, maxLength) {
  if (typeof value !== "string") return fallback;
  return value.trim().slice(0, maxLength) || fallback;
}

function cleanStringArray(value, maxItems = 5, maxLength = 100) {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item) => typeof item === "string")
    .slice(0, maxItems)
    .map((item) => item.trim().slice(0, maxLength))
    .filter(Boolean);
}

function cleanHistory(value) {
  if (!Array.isArray(value)) return [];

  return value
    .filter((message) =>
      message &&
      ["user", "assistant", "ai"].includes(message.role) &&
      typeof message.content === "string" &&
      message.content.trim().length > 0
    )
    .slice(-MAX_HISTORY_MESSAGES)
    .map((message) => ({
      // Le site utilise le rôle "ai". L'API Groq attend "assistant".
      role: message.role === "user" ? "user" : "assistant",
      content: message.content.trim().slice(0, MAX_HISTORY_MESSAGE_LENGTH)
    }));
}

function cleanImages(value) {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) {
    throw new Error("Le format des images est invalide.");
  }

  if (value.length > MAX_IMAGES) {
    throw new Error(`Tu peux joindre au maximum ${MAX_IMAGES} images par message.`);
  }

  const images = value;
  for (const image of images) {
    const validDataUrl =
      typeof image === "string" &&
      /^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/]+={0,2}$/.test(image);

    if (!validDataUrl) {
      throw new Error("Une image est invalide. Utilise JPG, PNG ou WEBP.");
    }
  }

  const totalChars = images.reduce((total, image) => total + image.length, 0);
  if (totalChars > MAX_IMAGES_TOTAL_CHARS) {
    throw new Error("Les images sont trop lourdes. Réduis leur taille puis réessaie.");
  }

  return images;
}

function getStyleInstruction(style) {
  const styles = {
    equilibre: "Donne une explication équilibrée, claire et de longueur raisonnable.",
    simple: "Réponds court et simplement. Va directement à l'essentiel.",
    pas_a_pas: "Détaille chaque étape de raisonnement pédagogique et chaque calcul utile.",
    exemples: "Ajoute des exemples concrets pour rendre les notions faciles à comprendre.",
    guide: "Commence par guider l'élève avec des indices. S'il demande la correction ou semble bloqué, donne ensuite la solution complète."
  };
  return styles[style] || styles.equilibre;
}

function getSystemPrompt(context) {
  const specialites = context.specialites.length
    ? context.specialites.join(", ")
    : "non précisées";

  return `Tu es FlowAI, un professeur particulier bienveillant qui aide des élèves.

Contexte de l'élève :
- Classe : ${context.niveau}
- Programme : ${context.programme}
- Voie : ${context.voie}
- Série ou filière : ${context.serie}
- Spécialités : ${specialites}
- Matière : ${context.matiere}

Consignes pédagogiques :
- Réponds en français, avec des mots simples adaptés au niveau indiqué.
- ${getStyleInstruction(context.style)}
- Pour un exercice, présente la méthode puis les étapes de résolution.
- Si l'élève demande directement la correction, donne-la avec les étapes nécessaires, sans cacher la réponse.
- Utilise un exemple concret quand cela aide. Si le programme est gabonais, privilégie si possible un contexte proche de la vie au Gabon.
- Si une image est jointe, observe-la attentivement. Recopie les éléments utiles de l'énoncé avant de résoudre.
- N'invente jamais les parties illisibles d'une image ou les données manquantes. Demande une photo plus nette si nécessaire.
- Si tu n'es pas certain d'une information, indique-le clairement.
- Reste bienveillant, encourageant et adapté à un élève. Ne juge jamais ses erreurs.
- Pour une question sans rapport avec les études, réponds brièvement si elle est simple, puis propose naturellement de revenir aux apprentissages.
- Utilise une mise en forme lisible : paragraphes courts, listes et formules si utiles.
- Pour toutes les expressions mathématiques, écris les formules en LaTeX : utilise \\( ... \\) pour une formule dans une phrase et \\[ ... \\] pour une formule centrée sur sa propre ligne. Exemple : \\( f(x)=x^2+3x-1 \\) et \\[ \\Delta=b^2-4ac \\].
- N'écris pas les formules importantes uniquement en texte brut (ex. x^2 ou sqrt(x)) : utilise les notations LaTeX afin qu'elles s'affichent proprement.
- Pour les calculs, garde les étapes sur des lignes séparées et encadre la réponse finale avec \\[ ... \\] lorsque c'est pertinent.
- Termine par une mini-vérification de compréhension uniquement lorsque cela paraît naturel.
- Utilise **texte** pour mettre les mots importants en gras.
- Utilise *texte* pour mettre certains mots en italique.
- Utilise les titres et les listes Markdown si nécessaire.
- Ne mets pas toute la réponse en gras.
- N'écris jamais ** ni * à l'intérieur d'une formule LaTeX. Pour mettre un résultat en valeur, mets le mot qui l'annonce en gras (par exemple **Résultat :**) puis écris la formule normalement.
- N'utilise pas de tableaux Markdown : présente les données avec des listes.`;
}

export async function onRequest(context) {
  const { request, env } = context;

  if (request.method !== "POST") {
    return json(405, { error: "Méthode non autorisée. Utilise POST." }, { Allow: "POST" });
  }

  if (!(await verifyUser(request, env))) {
    return json(401, { error: "Connexion requise" });
  }

  const apiKey = env.GROQ_API_KEY;
  if (!apiKey) {
    console.error("Configuration manquante : GROQ_API_KEY.");
    return json(500, {
      error: "Le service IA n'est pas configuré. L'administrateur doit vérifier la configuration."
    });
  }

  let rawBody;
  try {
    rawBody = await request.text();
  } catch {
    return json(400, { error: "La requête n'a pas pu être lue." });
  }
  if (rawBody.length > MAX_BODY_CHARS) {
    return json(413, { error: "La requête est trop volumineuse. Réduis la taille des images." });
  }

  let data;
  try {
    data = rawBody ? JSON.parse(rawBody) : {};
  } catch {
    return json(400, { error: "Le contenu de la requête n'est pas un JSON valide." });
  }

  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return json(400, { error: "Format de requête invalide." });
  }

  const question = cleanText(data.question, "", MAX_QUESTION_LENGTH);
  const context_ = {
    niveau: cleanText(data.niveau, "non précisé", 40),
    programme: cleanText(data.programme, "non précisé", 40),
    voie: cleanText(data.voie, "non précisée", 60),
    serie: cleanText(data.serie, "non précisée", 80),
    specialites: cleanStringArray(data.specialites),
    matiere: cleanText(data.matiere, "Toutes matières", 60),
    style: cleanText(data.style, "equilibre", 30)
  };

  if (!question) {
    return json(400, { error: "Écris une question avant d'envoyer ton message." });
  }

  let images;
  try {
    images = cleanImages(data.images);
  } catch (error) {
    return json(400, { error: error.message });
  }

  const historique = cleanHistory(data.historique);
  const messages = [
    { role: "system", content: getSystemPrompt(context_) },
    ...historique,
    {
      role: "user",
      content: images.length
        ? [
            { type: "text", text: question },
            ...images.map((url) => ({
              type: "image_url",
              image_url: { url }
            }))
          ]
        : question
    }
  ];

  const payload = {
    model: images.length
      ? env.GROQ_VISION_MODEL || DEFAULT_VISION_MODEL
      : env.GROQ_MODEL || DEFAULT_MODEL,
    temperature: 0.6,
    max_completion_tokens: MAX_OUTPUT_TOKENS,
    reasoning_effort: "low",
    messages
  };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    const responseText = await response.text();
    let result;
    try {
      result = responseText ? JSON.parse(responseText) : {};
    } catch {
      result = {};
    }

    if (!response.ok) {
      console.error("Erreur Groq :", response.status, JSON.stringify(result));

      if (response.status === 401 || response.status === 403) {
        return json(502, {
          error: "La connexion au service IA a échoué. L'administrateur doit vérifier la clé API Groq."
        });
      }
      if (response.status === 404) {
        return json(502, {
          error: "Le modèle IA demandé est introuvable. Vérifie les variables GROQ_MODEL et GROQ_VISION_MODEL dans Cloudflare."
        });
      }
      if (response.status === 413) {
        return json(413, { error: "La requête est trop volumineuse. Réduis la taille des images." });
      }
      if (response.status === 429) {
        return json(429, {
          error: "La limite de requêtes Groq est atteinte. Attends un peu avant de réessayer."
        }, {
          "Retry-After": response.headers.get("retry-after") || "30"
        });
      }
      if (response.status >= 500) {
        return json(503, {
          error: "Le service IA rencontre un problème temporaire. Réessaie dans quelques instants."
        });
      }
      return json(502, {
        error: "La demande n'a pas pu être traitée par le service IA."
      });
    }

    const answer = result?.choices?.[0]?.message?.content;
    if (typeof answer !== "string" || !answer.trim()) {
      console.error("Réponse Groq inattendue :", JSON.stringify(result));
      return json(502, { error: "Le service IA a renvoyé une réponse vide. Réessaie." });
    }

    return json(200, result);
  } catch (error) {
    if (error.name === "AbortError") {
      return json(504, { error: "La réponse met trop de temps à arriver. Réessaie." });
    }
    console.error("Erreur interne FlowAI :", error);
    return json(500, { error: "Une erreur interne est survenue. Réessaie dans quelques instants." });
  } finally {
    clearTimeout(timeout);
  }
}
