import time

import gradio as gr

from speed_compare.config import PROMPT_CHOICES, REPLAY_STEP_SECONDS, TASK_PROMPTS
from speed_compare.inference import prepare_image_for_prompt, run_diffusion_model, run_mineru_model
from speed_compare.parsers import make_diffusion_events, make_mineru_events
from speed_compare.preview_renderers import render_output_html, supports_render
from speed_compare.render import (
    APP_CSS,
    build_initial_view,
    build_rendered_panel,
    build_result_panel,
    build_waiting_panel,
)


def _log_generation_result(label: str, result: dict) -> None:
    print(f"\n===== {label} Result =====")
    print(result.get("response", "").strip() or "(empty response)")
    print(f"===== End {label} Result =====\n")


def _empty_compare_state() -> dict:
    return {
        "image_path": None,
        "prompt_type": "text",
        "mineru_result": None,
        "mineru_events": [],
        "mineru_duration": 0.0,
        "diffusion_result": None,
        "diffusion_events": [],
        "diffusion_duration": 0.0,
    }


def _toggle_button_update(prompt_type: str, enabled: bool, view_mode: str = "text"):
    if not supports_render(prompt_type):
        return gr.update(value="Render N/A", interactive=False)
    return gr.update(
        value="Show Text" if view_mode == "render" else "Show Render",
        interactive=enabled,
    )


def _panel_from_state(compare_state: dict, model_key: str, view_mode: str) -> str:
    if model_key == "mineru":
        title = "MinerU 2.5"
        result = compare_state["mineru_result"]
        events = compare_state["mineru_events"]
        duration = float(compare_state["mineru_duration"] or 0.0)
        mode = "append"
    else:
        title = "MinerU-Diffusion"
        result = compare_state["diffusion_result"]
        events = compare_state["diffusion_events"]
        duration = float(compare_state["diffusion_duration"] or 0.0)
        mode = "positional"

    if not result:
        initial_mineru, initial_diffusion = build_initial_view()
        return initial_mineru if model_key == "mineru" else initial_diffusion

    if view_mode == "render" and supports_render(compare_state["prompt_type"]):
        rendered_output = render_output_html(
            compare_state["prompt_type"],
            result.get("response", ""),
            compare_state.get("image_path"),
        )
        return build_rendered_panel(title, result, duration, "Done!", rendered_output)

    return build_result_panel(title, result, events, duration, mode, duration, "Done!")


def toggle_render_view(model_key: str, compare_state: dict, view_mode: str):
    if not compare_state or not compare_state.get(f"{model_key}_result"):
        return (
            _panel_from_state(compare_state or _empty_compare_state(), model_key, "text"),
            "text",
            _toggle_button_update("text", False, view_mode),
        )

    prompt_type = compare_state.get("prompt_type", "text")
    if not supports_render(prompt_type):
        return (
            _panel_from_state(compare_state, model_key, "text"),
            "text",
            _toggle_button_update(prompt_type, False, "text"),
        )

    next_view_mode = "render" if view_mode != "render" else "text"
    return (
        _panel_from_state(compare_state, model_key, next_view_mode),
        next_view_mode,
        _toggle_button_update(prompt_type, True, next_view_mode),
    )


def run_compare(image_path: str | None, prompt_type: str):
    initial_mineru, initial_diffusion = build_initial_view()
    empty_state = _empty_compare_state()

    if not image_path:
        yield (
            initial_mineru,
            initial_diffusion,
            empty_state,
            "text",
            "text",
            _toggle_button_update(prompt_type, False, "text"),
            _toggle_button_update(prompt_type, False, "text"),
        )
        return

    if prompt_type not in TASK_PROMPTS:
        yield (
            initial_mineru,
            initial_diffusion,
            empty_state,
            "text",
            "text",
            _toggle_button_update("text", False, "text"),
            _toggle_button_update("text", False, "text"),
        )
        return

    yield (
        build_waiting_panel("MinerU 2.5", "Running", "append"),
        build_waiting_panel("MinerU-Diffusion", "Running", "positional"),
        empty_state,
        "text",
        "text",
        _toggle_button_update(prompt_type, False, "text"),
        _toggle_button_update(prompt_type, False, "text"),
    )

    try:
        prepared_image_path = str(prepare_image_for_prompt(image_path, prompt_type))
        mineru_result = run_mineru_model(prepared_image_path, prompt_type, assume_prepared=True)
        mineru_events, mineru_duration = make_mineru_events(mineru_result)
        diffusion_result = run_diffusion_model(prepared_image_path, prompt_type, assume_prepared=True)
        _log_generation_result("MinerU 2.5", mineru_result)
        _log_generation_result("MinerU-Diffusion", diffusion_result)
    except Exception as exc:
        print(exc)
        yield (
            initial_mineru,
            initial_diffusion,
            empty_state,
            "text",
            "text",
            _toggle_button_update(prompt_type, False, "text"),
            _toggle_button_update(prompt_type, False, "text"),
        )
        return

    diffusion_events, diffusion_duration = make_diffusion_events(diffusion_result)
    max_time = max(mineru_duration, diffusion_duration, 0.0)
    compare_state = {
        "image_path": prepared_image_path,
        "prompt_type": prompt_type,
        "mineru_result": mineru_result,
        "mineru_events": mineru_events,
        "mineru_duration": mineru_duration,
        "diffusion_result": diffusion_result,
        "diffusion_events": diffusion_events,
        "diffusion_duration": diffusion_duration,
    }

    yield (
        build_result_panel("MinerU 2.5", mineru_result, mineru_events, mineru_duration, "append", 0.0, "Working"),
        build_result_panel(
            "MinerU-Diffusion",
            diffusion_result,
            diffusion_events,
            diffusion_duration,
            "positional",
            0.0,
            "Working",
        ),
        empty_state,
        "text",
        "text",
        _toggle_button_update(prompt_type, False, "text"),
        _toggle_button_update(prompt_type, False, "text"),
    )

    replay_start = time.perf_counter()
    last_replay_time = -1.0
    while True:
        replay_time = min(time.perf_counter() - replay_start, max_time)
        if replay_time <= last_replay_time and replay_time < max_time:
            time.sleep(REPLAY_STEP_SECONDS)
            continue
        last_replay_time = replay_time
        mineru_status = "Done!" if replay_time >= mineru_duration else "Working"
        diffusion_status = "Done!" if replay_time >= diffusion_duration else "Working"

        yield (
            build_result_panel(
                "MinerU 2.5",
                mineru_result,
                mineru_events,
                mineru_duration,
                "append",
                replay_time,
                mineru_status,
            ),
            build_result_panel(
                "MinerU-Diffusion",
                diffusion_result,
                diffusion_events,
                diffusion_duration,
                "positional",
                replay_time,
                diffusion_status,
            ),
            empty_state,
            "text",
            "text",
            _toggle_button_update(prompt_type, False, "text"),
            _toggle_button_update(prompt_type, False, "text"),
        )
        if replay_time >= max_time:
            break
        if max_time - replay_time > 0:
            time.sleep(REPLAY_STEP_SECONDS)

    yield (
        build_result_panel(
            "MinerU 2.5",
            mineru_result,
            mineru_events,
            mineru_duration,
            "append",
            mineru_duration,
            "Done!",
        ),
        build_result_panel(
            "MinerU-Diffusion",
            diffusion_result,
            diffusion_events,
            diffusion_duration,
            "positional",
            diffusion_duration,
            "Done!",
        ),
        compare_state,
        "text",
        "text",
        _toggle_button_update(prompt_type, True, "text"),
        _toggle_button_update(prompt_type, True, "text"),
    )


def build_demo() -> gr.Blocks:
    initial_mineru, initial_diffusion = build_initial_view()

    with gr.Blocks(title="MinerU Speed Compare") as demo:
        gr.Markdown("## MinerU Speed Compare", elem_classes=["panel-title"])
        with gr.Row(elem_id="app-shell"):
            with gr.Column(scale=5, elem_id="left-col"):
                image_input = gr.Image(label="Input Image", type="filepath", sources=["upload"])
                prompt_input = gr.Radio(label="OCR Type", choices=PROMPT_CHOICES, value="text", elem_id="prompt-grid")
                run_button = gr.Button("Start Comparison", variant="primary", elem_id="start-button")

            with gr.Column(scale=6, elem_id="mineru-col"):
                mineru_panel = gr.HTML(initial_mineru)
                mineru_render_button = gr.Button("Render N/A", interactive=False, elem_id="mineru-render-toggle")

            with gr.Column(scale=6, elem_id="diffusion-col"):
                diffusion_panel = gr.HTML(initial_diffusion)
                diffusion_render_button = gr.Button("Render N/A", interactive=False, elem_id="diffusion-render-toggle")

        compare_state = gr.State(_empty_compare_state())
        mineru_view_mode = gr.State("text")
        diffusion_view_mode = gr.State("text")

        run_button.click(
            fn=run_compare,
            inputs=[image_input, prompt_input],
            outputs=[
                mineru_panel,
                diffusion_panel,
                compare_state,
                mineru_view_mode,
                diffusion_view_mode,
                mineru_render_button,
                diffusion_render_button,
            ],
            show_progress="hidden",
        )

        mineru_render_button.click(
            fn=lambda state, view_mode: toggle_render_view("mineru", state, view_mode),
            inputs=[compare_state, mineru_view_mode],
            outputs=[mineru_panel, mineru_view_mode, mineru_render_button],
            show_progress="hidden",
        )

        diffusion_render_button.click(
            fn=lambda state, view_mode: toggle_render_view("diffusion", state, view_mode),
            inputs=[compare_state, diffusion_view_mode],
            outputs=[diffusion_panel, diffusion_view_mode, diffusion_render_button],
            show_progress="hidden",
        )

    demo.queue(default_concurrency_limit=1)
    return demo


demo = build_demo()


def launch() -> None:
    demo.launch(server_name="0.0.0.0", css=APP_CSS)
