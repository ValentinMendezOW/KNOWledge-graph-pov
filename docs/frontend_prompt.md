# Lovable Prompt

Build a React frontend for an internal consultant-facing knowledge search tool called `Consulting Research Explorer`.

Product goal:
- Let consultants search an internal corpus of documents with citation-grounded answers.
- Show exactly which source documents and excerpts support the answer.
- Make the product feel like a serious research assistant, not a chat toy.

Users:
- Consultants at all levels.
- Internal pilot only.

Current backend reality:
- The backend already indexes documents and returns an answer plus ranked citations.
- Source metadata includes:
  - `title`
  - `organization`
  - `published_date`
  - `file_name`
  - `access_tier`
  - `practices`
  - `industries`
  - `topics`
  - `excerpt`
- Access model for now is simple:
  - `all_access`
  - `limited_access`
- In the current pilot there may be zero restricted documents, so the UI must handle that gracefully.

Design direction:
- Clean, premium, research-heavy interface.
- Light mode first.
- Avoid generic chatbot styling.
- Use a strong editorial layout with a left search/navigation rail, a central answer pane, and a right citation/source pane on desktop.
- On mobile, stack these areas cleanly.
- Visual tone: knowledge platform for strategy consultants.
- Prefer warm neutrals, sharp typography, subtle data-dashboard cues, and restrained motion.

Core screens:
1. Search workspace
2. Source detail drawer or panel
3. Index status/admin panel

Search workspace requirements:
- Top search bar for natural-language questions.
- Search submit button.
- Optional filters for organization, practice, industry, topic, and year.
- Show the generated answer in the main pane.
- Show citations as numbered source cards.
- Clicking a citation opens the source detail panel.
- Each source card should show:
  - title
  - organization
  - published date
  - access tier
  - short excerpt
  - tags for practice, industry, topic
- Include a visible trust pattern:
  - “Answer grounded in X sources”
  - source timestamps
  - citation numbers inline

Index status/admin panel requirements:
- Show corpus size
- Show chunk count
- Show whether OpenAI is configured
- Show current Neo4j target
- Show whether restricted documents are configured
- Include actions styled as buttons for:
  - rebuild index
  - load into Neo4j

Interaction model:
- Use mock JSON data and mock async APIs for now.
- Structure the app so the API layer can later be replaced with real endpoints.
- Support loading, empty, and error states.
- Keep components reusable and well factored.

Suggested frontend architecture:
- React + TypeScript
- Tailwind CSS
- shadcn/ui or similarly composable component primitives
- TanStack Query for async data
- Zustand or lightweight local state only if actually needed

Mock backend contracts:
- `POST /api/search`
  - request:
    - `question: string`
    - `access_mode: "all_access" | "limited_access"`
    - `allowed_document_ids: string[]`
  - response:
    - `answer: string`
    - `used_llm: boolean`
    - `hits: Array<{ score: number, document: { doc_id: string, title: string, organization: string, published_date: string | null, file_name: string, access_tier: string, practices: string[], industries: string[], topics: string[] }, chunk: { chunk_id: string, chunk_index: number }, excerpt: string }>`

- `GET /api/status`
  - response:
    - `document_count: number`
    - `chunk_count: number`
    - `openai_configured: boolean`
    - `neo4j_uri: string`
    - `restricted_document_count: number`

Deliverables:
- A polished search page
- Reusable components
- Mock data layer
- Responsive layout
- Empty/loading/error states
- A small README section explaining the component structure and how to swap mocks for real APIs
