# Pat 剪刀 · AI editing skill for Chinese talking-head videos

[中文](README.md) | English

A skill for AI coding assistants (Claude Code, TRAE) that turns a Mandarin talking-head clip shot on a phone, or a voice-only recording, into a finished video with Chinese and English subtitles. You say one line, “剪成成片” (“cut it into a finished video”), and it stops twice for your approval: once for the list of cuts, once for the rough cut.

**Chinese (Mandarin) speech only.** The speech-recognition model, the filler-word rules and the subtitle rules are all built for Chinese. English words inside Chinese speech are fine; an English talking-head is not supported. The skill and the docs are written in Chinese. It needs an Apple silicon Mac (M1 or later).

剪刀 (jiǎndāo) means scissors. It is item 1 of 文科生 Pat 的文具盒 skills, a liberal-arts student's pencil case of skills.

## Demo

**Before: the raw clip, 44 seconds** (shot casually on a phone with no script: one “嗯”, one “哎”, two sentences said twice, a few pauses of one or two seconds)

https://github.com/user-attachments/assets/8aef709a-3e0c-43b1-9899-e1912198b8be

**After: 25 seconds** (vertical, full-frame face, Chinese and English subtitles, tight pacing, no music. The author approved the cut list and the rough cut; the subtitles were transcribed by the AI from the edited audio. The “哎” at 0:07 runs straight into the next word, so it could not be cut cleanly and was kept.)

https://github.com/user-attachments/assets/61054444-e206-4048-922b-2cd54b87fff4

## What it does

- Removes filler words, false starts and repeated takes (keeping the last complete one) and shortens long pauses. After cutting, it transcribes the edit again and checks it word by word against the approved plan.
- Landscape 16:9 explainer: given a script, it finds one visual per sentence (a screenshot of the source web page with that sentence boxed, or Pixabay stock), adds a round face cam in the lower left, transitions and background music.
- Vertical 9:16: full-frame face by default.
- Writes an .srt subtitle file, and can export the rough cut as a draft for 剪映 (the Chinese edition of CapCut). The draft export is unofficial and was tested only with 剪映专业版 9.6.0 on Mac.
- Transcription runs locally with FunASR. Your AI assistant reads the transcript, the cut plan and some screenshots, and those go to the assistant's servers.

## Requirements

- An Apple silicon Mac. Intel Macs are untested; Windows is not supported.
- About 4.3 GB once installed; the environment check asks for at least 10 GB free.
- Claude Code (the whole workflow was tested; needs a Claude subscription or API credits), or TRAE China edition (partly tested). Other assistants are untested.
- Optional: Google Chrome (web screenshots), a free Pixabay API key (stock footage), 剪映专业版 (draft export).

## Install

The step-by-step guide is in Chinese: [docs/安装说明.md](docs/安装说明.md). In short: get the code (Code → Download ZIP, rename the folder to `pat-jiandao-skill`, or `git clone`), open the folder in Claude Code and ask it to install by following `docs/安装说明.md`, then run the environment check. It lists anything missing and how to install it:

```bash
uv run python scripts/doctor.py
```

Before recording on an iPhone, turn off Settings → Camera → Record Video → HDR Video.

## License

The code is under the [MIT License](LICENSE). The bundled font, and the models and libraries downloaded during install, keep their own licenses: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This project is not affiliated with or endorsed by Anthropic (Claude, Claude Code), ByteDance (TRAE, 剪映, CapCut), or any other company or project mentioned here. It is maintained by one person, so replies may be slow.
