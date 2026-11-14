  # Frontend contract

  The API base URL is `http://127.0.0.1:8000` in local dev. CORS allows `http://localhost:3000`.

  List sessions with `GET /sessions`. Open one with `GET /sessions/{session_id}` (transcript + phase). Continue chatting via `POST /chat` with that `session_id`. Uploads still use `POST /uploads` and `POST /uploads/complete`.

  ## Endpoints

  ### `GET /health`

  Returns `{ "status": "healthy" }`.

  ### `GET /sessions`

  Returns recent sessions, newest first:

  ```json
  {
    "sessions": [
      {
        "session_id": "uuid",
        "created_at": 1710000000,
        "updated_at": 1710000100,
        "preview": "Check this squat."
      }
    ]
  }
  ```

  `preview` is the first user message (truncated), or `null` when the session has none yet.

  ### `GET /sessions/{session_id}`

  Opens one chatbox. Unknown `session_id` is HTTP 404.

  ```json
  {
    "session_id": "uuid",
    "created_at": 1710000000,
    "updated_at": 1710000100,
    "phase": "done",
    "progress": null,
    "messages": [
      { "id": "…", "kind": "text", "role": "user", "text": "Check this squat.", "video_urls": ["https://..."] },
      {
        "id": "job-uuid",
        "kind": "analysis",
        "job": {
          "job_id": "job-uuid",
          "status": "done",
          "error": null,
          "reps": {},
          "series": [],
          "overlay_url": "https://...",
          "image_urls": [],
          "reply": "plain-English coach assessment"
        }
      },
      { "id": "…", "kind": "text", "role": "assistant", "text": "plain-English coach assessment" }
    ]
  }
  ```

  `phase` is `idle`, `queued`, `running`, `coaching`, `done`, or `failed`. While `running`, `progress` may be `{ "frame": 12, "frame_count": 240 }`. Render `messages` in order: text bubbles and analysis cards (same shape as a finished-job `ready` event). System and tool-call-only rows are omitted. A user upload shows up as a text message whose `video_urls` are presigned GET links for the original clip (expire in 1 hour; reload the session for fresh links). Fresh `overlay_url` / `image_urls` are presigned the same way. If `phase` is `queued`, `running`, or `coaching`, call `POST /chat` with `{ "session_id", "text": "" }` to resume.

  ### `POST /chat`

  Body:

  ```json
  { "session_id": "optional-existing-id", "text": "Check this squat." }
  ```

  Omit `session_id` to start a new chat. A new chat always means a new session: the server creates one and the stream's first event is `session`:

  ```
  event: session
  data: { "session_id": "uuid" }
  ```

  Send `Accept: text/event-stream`. An unknown `session_id` is HTTP 404. `text` may be `""`. The body stays open until a `ready` or `error` event.

  You can start chatting with plain text before ever uploading a video — the model just chats. Once a video is queued for analysis (see `/uploads/complete`), `/chat` blocks on that job: it streams progress and the coach assessment instead of starting a new turn. Any non-empty `text` is still saved on the session (before the analysis tool messages) so notes sent with the upload are kept. The user can't do anything else with the session until that job finishes. Keep calling `/chat` (with the note, or `text: ""`) to resume watching a job that's still running.

  ### `POST /uploads`

  Body:

  ```json
  { "session_id": "optional-existing-id", "content_type": "video/mp4" }
  ```

  `content_type` is optional and defaults to `video/mp4`. Allowed values:

  | `content_type` | Extension | Typical source |
  | --- | --- | --- |
  | `video/mp4` | `.mp4` | phone and camera clips |
  | `video/quicktime` | `.mov` | iPhone |
  | `video/webm` | `.webm` | browser recordings |

  Anything else is HTTP 422. Omit `session_id` to create a session (e.g. to upload before chatting at all). Pass an existing `session_id` to upload another video into that same conversation whenever you want — before, during, or after a chat. An unknown `session_id` is HTTP 404.

  Response:

  ```json
  {
    "session_id": "uuid",
    "upload_url": "https://...",
    "key": "uploads/{session_id}/{hex}.mp4",
    "content_type": "video/mp4"
  }
  ```

  `PUT` the file bytes to `upload_url` with the returned `content_type` (the signature requires that exact header). The URL expires in 1 hour. Do not send the video to the API. The file picker should accept `.mp4`, `.mov`, and `.webm`.

  ### `POST /uploads/complete`

  Body:

  ```json
  { "session_id": "uuid", "key": "uploads/{session_id}/{hex}.mp4" }
  ```

  `key` is the value returned by `POST /uploads` (`.mp4`, `.mov`, or `.webm`).

  Call this once the `PUT` to `upload_url` finishes. It immediately queues analysis — no chat message or tool call is needed to start it. An unknown `session_id` is HTTP 404.

  Response:

  ```json
  { "job_id": "uuid" }
  ```

  Call `/chat` next (any `text`, including `""`) to watch it run and receive the coaching result — see above.

  ## Client state machine

  Persist `session_id`. Drive the UI from `idle`, `chatting`, `uploading`, `queued`, `running`, `coaching`, `done`, and `failed`.

  ```mermaid
  stateDiagram-v2
      [*] --> idle
      idle --> chatting: POST_chat_no_session_id
      chatting --> chatting: POST_chat_with_text
      chatting --> uploading: POST_uploads
      uploading --> queued: PUT_video_then_POST_uploads_complete
      queued --> running: job_status_running
      running --> coaching: job_status_done
      running --> failed: job_status_failed
      coaching --> done: ready_with_reply
      done --> chatting: POST_chat_with_text
      done --> uploading: POST_uploads_again
      failed --> uploading: POST_uploads_again
  ```

  Typical first run:

  1. `POST /chat` with `{ "text": "Hi, planning to film a squat set." }` and no `session_id`. Read the `session` event to get `session_id`, then append `delta` text until `ready`.
  2. `POST /uploads` with `{ "session_id": ..., "content_type": "video/mp4" }` (or `video/quicktime` / `video/webm`).
  3. `PUT` the file to `upload_url` with that `Content-Type`.
  4. `POST /uploads/complete` with `{ "session_id": ..., "key": ... }`.
  5. `POST /chat` with `{ "session_id": ..., "text": "" }`. Show progress from `job` events, then append coach `delta` text as it streams. The single `ready` at the end carries the full results payload (same shape as before) plus the coach `reply`.
  6. Continue chatting normally; `/chat` behaves like a plain turn again now that the job has a stored reply.

  To check another video, repeat steps 2-5 with the same `session_id` at any point, including mid-conversation. The next `/chat` call will block on the new job the same way.

  ## SSE events

  Parse `event:` and `data:` lines. A blank line ends one event. Ignore comment lines that start with `:`, including the `: ping` keepalive sent about every 0.5s while nothing has changed.

  ### `session` (first `/chat` event, only when `session_id` was omitted)

  ```json
  { "session_id": "uuid" }
  ```

  ### `delta`

  ```json
  { "text": "partial assistant text" }
  ```

  Append `text` to the current assistant bubble. Sent for both plain chat replies and the streamed coach assessment.

  ### `job`

  ```json
  {
    "job_id": "uuid",
    "status": "running",
    "progress": { "frame": 12, "frame_count": 240 },
    "error": null
  }
  ```

  `status` is `queued`, `running`, `done`, or `failed`. While `running`, show `frame / frame_count`.

  ### `ready`

  Ends the `/chat` response. A plain turn with no job involved:

  ```json
  { "status": "done" }
  ```

  A failed job:

  ```json
  { "status": "failed", "error": "..." }
  ```

  A finished job (after the coach `delta`s) uses the results payload below.

  ### `error`

  ```json
  { "detail": "Unknown session" }
  ```

  Stop the stream and show `detail`.

  ## Results payload

  `ready` once a job's coaching finishes:

  ```json
  {
    "job_id": "uuid",
    "status": "done",
    "error": null,
    "reps": {},
    "series": [],
    "overlay_url": "https://...",
    "image_urls": [],
    "reply": "plain-English coach assessment"
  }
  ```

  `overlay_url` is the skeleton video. `image_urls` are presigned PNGs, sorted by filename. Four stills per rep when that frame exists: `*_rep{N}_standing.png`, `*_rep{N}_down.png` (descent midpoint), `*_rep{N}_seated.png` (bottom), and `*_rep{N}_drive.png`. Pair them by the `rep{N}` token and the event suffix in the object key inside the URL. The coach sees only rep 1's stills.

  `reps` is the analysis object:

  ```json
  {
    "video": "input.mp4",
    "model": "rtmpose",
    "fps": 30.0,
    "side": "left",
    "reason": "success",
    "reps": [
      {
        "rep": 1,
        "down_ms": 800,
        "bottom_ms": 200,
        "up_ms": 700,
        "standing": {},
        "bottom": {},
        "drive": {},
        "finish": {}
      }
    ]
  }
  ```

  `drive` and `finish` may be `null`. `up_ms` may be `null` when there is no finish. Each snapshot is:

  ```json
  {
    "frame": 12,
    "t_ms": 400,
    "confidence": 0.91,
    "knee_deg": 95.2,
    "hip_deg": 80.1,
    "back_from_vertical_deg": 30.4,
    "hip_below_knee": true,
    "hip_knee_gap": 0.04,
    "knee_forward": 0.12,
    "nose_below_ear": 0.01
  }
  ```

  `side` is `left`, `right`, or `null`.

  `reason` is `success` when at least one rep was found. Otherwise it says why segmentation produced none, for example `not enough confident knee angles to segment reps` or `no knee-angle valley deep enough to mark a squat`.

  `series` is one point per sampled frame of the chosen side, for the chart only. It is stored on the tool message for that job, so each analysis keeps its own series. It is not part of the coach text.

  ```json
  {
    "t_ms": 0,
    "knee_deg": 170.0,
    "hip_deg": 175.0,
    "back_from_vertical_deg": 5.0
  }
  ```

  Angles may be `null` on a frame the pose model missed.

  Only the latest job per session is used by `/chat`. Older jobs stay in the database and are not listed, but their results remain in the chat history.

  ## Clicking the results box

  Show a results card once `ready.status` is `done` and `reply` is present. Keep the card after that.

  - Play `overlay_url`.
  - Chart `series` with `t_ms` on the x-axis and `knee_deg`, `hip_deg`, and `back_from_vertical_deg` on the y-axis. Skip null angles.
  - One row per item in `reps.reps`: rep number, descent (`down_ms`), pause (`bottom_ms`), and ascent (`up_ms`).
  - Clicking a rep seeks the overlay to `standing.t_ms`, shades the chart from `standing.t_ms` through `finish.t_ms` (or `bottom.t_ms` when `finish` is null), and shows that rep's four stills. Skip a still that was not produced. Under each still, show plain tags and skip a tag whose inputs are missing:

    - Standing: `nose_below_ear` above zero is "Looking at the floor", otherwise "Looking up". `back_from_vertical_deg` above 15 is "Already leaned", otherwise "Chest up".
    - Down: change in `back_from_vertical_deg` from standing to bottom. Under 8 is "Chest stayed up", 8 through 25 is "Leaned into the hole", above 25 is "Folded forward". Absolute change in `knee_forward` of at least 0.15 is "Knees came forward", otherwise "Knees stayed back".
    - Seated: at the bottom, `hip_below_knee` with `hip_knee_gap` above 0.1 is "Below parallel". A gap within 0.1 of zero is "Around parallel". Otherwise "Above parallel".
    - Drive up: from bottom to drive, a more horizontal back with an opening hip is "Hips rose first". A more vertical back while the hip stays closed is "Chest lifted early". Otherwise "Hips and chest rose together".

  On `failed`, show `error` and offer a new upload. Do not render images or the chart.

  URLs on the card expire after 1 hour.
