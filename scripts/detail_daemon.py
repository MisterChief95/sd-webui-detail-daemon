import os
import gradio as gr
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import modules.scripts as scripts
from modules.infotext_utils import PasteField
from modules.script_callbacks import on_cfg_denoiser, remove_callbacks_for_function, on_infotext_pasted
from modules.ui_components import InputAccordion


def parse_infotext(infotext, params):
    if 'Detail Daemon' not in params:
        return
    
    try:
        d = {}
        for s in params['Detail Daemon'].split(','):
            k, _, v = s.partition(':')
            d[k.strip()] = v.strip()
        params['Detail Daemon'] = d
    except Exception:
        pass


on_infotext_pasted(parse_infotext)


def extract_infotext(d: dict, *keys: str):
    for key in keys:
        if key in d:
            return d[key]
    return None


def find_first_attr(obj, default, *attr: str):
    for a in attr:
        if hasattr(obj, a):
            return getattr(obj, a)
    return default


class Script(scripts.Script):

    FIELD_NAMES = {
        "enabled": ("DD ","dd_"),
        "mode": ("DD Mode",'mode', 'DD_mode', "dd_mode"),
        "amount": ("DD Amount",'amount', 'DD_amount', "dd_amount"),
        "start": ("DD Start",'st', 'DD_start', "dd_start"),
        "end": ("DD End",'ed', 'DD_end', "dd_end"),
        "bias": ("DD Bias",'bias', 'DD_bias', "dd_bias"),
        "exponent": ("DD Exponent",'exp', 'DD_exponent', "dd_exponent"),
        "start_offset": ("DD Start Offset",'st_offset', 'DD_start_offset', "dd_start_offset"),
        "end_offset": ("DD End Offset",'ed_offset', 'DD_end_offset', "dd_end_offset"),
        "fade": ("DD Fade",'fade', 'DD_fade', "dd_fade"),
        "smooth": ("DD Smooth",'smooth', 'DD_smooth', "dd_smooth"),
    }

    def __init__(self):
        super().__init__()
        self.schedule_params: dict[str, float] = None
        self.schedule = None

    def title(self):
        return "Detail Daemon"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label="Detail Daemon", elem_id=self.elem_id('detail-daemon')) as gr_enabled:
            with gr.Row():
                with gr.Column(scale=2):                    
                    gr_amount_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.10, label="Detail Amount")
                    gr_start = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.2, label="Start")
                    gr_end = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.8, label="End") 
                    gr_bias = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.5, label="Bias")                                                                                                                          
                with gr.Column(scale=1, min_width=275):  
                    preview = self.visualize(False, 0.2, 0.8, 0.5, 0.1, 1, 0, 0, 0, True)                                 
                    gr_vis = gr.Plot(value=preview, elem_classes=['detail-daemon-vis'], show_label=False)
            with gr.Accordion("More Knobs:", elem_classes=['detail-daemon-more-accordion'], open=False):
                with gr.Row():
                    with gr.Column(scale=2):   
                        with gr.Row():                                              
                            gr_start_offset_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.00, label="Start Offset", min_width=60) 
                            gr_end_offset_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.00, label="End Offset", min_width=60) 
                        with gr.Row():
                            gr_exponent = gr.Slider(minimum=0.0, maximum=10.0, step=.05, value=1.0, label="Exponent", min_width=60) 
                            gr_fade = gr.Slider(minimum=0.0, maximum=1.0, step=.05, value=0.0, label="Fade", min_width=60) 
                        # Because the slider max and min are sometimes too limiting:
                        with gr.Row():
                            gr_amount = gr.Number(value=0.10, precision=4, step=.01, label="Amount", min_width=60)  
                            gr_start_offset = gr.Number(value=0.0, precision=4, step=.01, label="Start Offset", min_width=60)  
                            gr_end_offset = gr.Number(value=0.0, precision=4, step=.01, label="End Offset", min_width=60) 
                    with gr.Column(scale=1, min_width=275): 
                        gr_mode = gr.Dropdown(["both", "cond", "uncond"], value="uncond", label="Mode", show_label=True, min_width=60, elem_classes=['detail-daemon-mode']) 
                        gr_smooth = gr.Checkbox(label="Smooth", value=True, min_width=60, elem_classes=['detail-daemon-smooth'])
                        gr.Markdown("## [Ⓗ Help](https://github.com/muerrilla/sd-webui-detail-daemon)", elem_classes=['detail-daemon-help'])

        gr_amount_slider.release(None, gr_amount_slider, gr_amount, _js="(x) => x")
        gr_amount.change(None, gr_amount, gr_amount_slider, _js="(x) => x")

        gr_start_offset_slider.release(None, gr_start_offset_slider, gr_start_offset, _js="(x) => x")
        gr_start_offset.change(None, gr_start_offset, gr_start_offset_slider, _js="(x) => x")

        gr_end_offset_slider.release(None, gr_end_offset_slider, gr_end_offset, _js="(x) => x")
        gr_end_offset.change(None, gr_end_offset, gr_end_offset_slider, _js="(x) => x")

        vis_args = [gr_enabled, gr_start, gr_end, gr_bias, gr_amount, gr_exponent, gr_start_offset, gr_end_offset, gr_fade, gr_smooth]
        for vis_arg in vis_args:
            if isinstance(vis_arg, gr.components.Slider):
                vis_arg.release(fn=self.visualize, show_progress=False, inputs=vis_args, outputs=[gr_vis])
            else:
                vis_arg.change(fn=self.visualize, show_progress=False, inputs=vis_args, outputs=[gr_vis])

        self.infotext_fields = [
            PasteField(gr_enabled, lambda d: any(any(field in key for field in self.FIELD_NAMES["enabled"]) for key in d.keys()), api="dd_enabled"),
            PasteField(gr_mode, lambda d: extract_infotext(d, *self.FIELD_NAMES["mode"]), api="dd_mode"),
            PasteField(gr_amount, lambda d: extract_infotext(d, *self.FIELD_NAMES["amount"]), api="dd_amount"),
            PasteField(gr_start, lambda d: extract_infotext(d, *self.FIELD_NAMES["start"]), api="dd_start"),
            PasteField(gr_end, lambda d: extract_infotext(d, *self.FIELD_NAMES["end"]), api="dd_end"),
            PasteField(gr_bias, lambda d: extract_infotext(d, *self.FIELD_NAMES["bias"]), api="dd_bias"),
            PasteField(gr_exponent, lambda d: extract_infotext(d, *self.FIELD_NAMES["exponent"]), api="dd_exponent"),
            PasteField(gr_start_offset, lambda d: extract_infotext(d, *self.FIELD_NAMES["start_offset"]), api="dd_start_offset"),
            PasteField(gr_end_offset, lambda d: extract_infotext(d, *self.FIELD_NAMES["end_offset"]), api="dd_end_offset"),
            PasteField(gr_fade, lambda d: extract_infotext(d, *self.FIELD_NAMES["fade"]), api="dd_fade"),
            PasteField(gr_smooth, lambda d: bool(int(extract_infotext(d, *self.FIELD_NAMES["smooth"]))), api="dd_smooth"),
        ]
        return [gr_enabled, gr_mode, gr_start, gr_end, gr_bias, gr_amount, gr_exponent, gr_start_offset, gr_end_offset, gr_fade, gr_smooth]
    
    def process(self, p, enabled, mode, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):    
        enabled = find_first_attr(p, enabled, *self.FIELD_NAMES["enabled"])
        mode = find_first_attr(p, mode, *self.FIELD_NAMES["mode"])
        amount = find_first_attr(p, amount, *self.FIELD_NAMES["amount"])
        start = find_first_attr(p, start, *self.FIELD_NAMES["start"])
        end = find_first_attr(p, end, *self.FIELD_NAMES["end"])
        bias = find_first_attr(p, bias, *self.FIELD_NAMES["bias"])
        exponent = find_first_attr(p, exponent, *self.FIELD_NAMES["exponent"])
        start_offset = find_first_attr(p, start_offset, *self.FIELD_NAMES["start_offset"])
        end_offset = find_first_attr(p, end_offset, *self.FIELD_NAMES["end_offset"])
        fade = find_first_attr(p, fade, *self.FIELD_NAMES["fade"])
        smooth = find_first_attr(p, smooth, *self.FIELD_NAMES["smooth"])

        if enabled:
            if p.sampler_name in ["DPM adaptive", "HeunPP2"]:
                tqdm.write(f'\033[33mWARNING:\033[0m Detail Daemon does not work with {p.sampler_name}')
                return
            
            self.schedule_params = {
                "start": start,
                "end": end,
                "bias": bias,
                "amount": amount,
                "exponent": exponent,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "fade": fade,
                "smooth": smooth
            }
            self.mode = mode
            self.cfg_scale = p.cfg_scale
            self.batch_size = p.batch_size
            on_cfg_denoiser(self.denoiser_callback)              
            self.callback_added = True 
            p.extra_generation_params.update({
                'DD Mode': mode,
                'DD Amount': amount,
                'DD Start': start,
                'DD End': end,
                'DD Bias': bias,
                'DD Exponent': exponent,
                'DD Start Offset': start_offset,
                'DD End Offset': end_offset,
                'DD Fade': fade,
                'DD Smooth': smooth
            })
            tqdm.write('\033[32mINFO:\033[0m Detail Daemon is enabled')
        else:
            if hasattr(self, 'callback_added'):
                remove_callbacks_for_function(self.denoiser_callback)
                delattr(self, 'callback_added')
                self.schedule = None
                # tqdm.write('\033[90mINFO: Detail Daemon callback removed\033[0m')  

    def before_process_batch(self, p, *args, **kwargs):
        self.is_hires = False

    def postprocess(self, p, processed, *args):
        if hasattr(self, 'callback_added'):
            remove_callbacks_for_function(self.denoiser_callback)
            delattr(self, 'callback_added')
            self.schedule = None
            # tqdm.write('\033[90mINFO: Detail Daemon callback removed\033[0m')

    def before_hr(self, p, *args):
        self.is_hires = True
        enabled = args[0]
        if enabled:
            tqdm.write(f'\033[33mINFO:\033[0m Detail Daemon does not work during Hires Fix')
        
    def denoiser_callback(self, params): 
        if self.is_hires:
            return
        step = max(params.sampling_step, params.denoiser.step)
        total_steps = max(params.total_sampling_steps, params.denoiser.total_steps)
        corrected_step_count = total_steps - max(total_steps // params.denoiser.steps - 1, 0)
        if self.schedule is None:
            self.schedule = self.make_schedule(corrected_step_count, **self.schedule_params)

        idx = min(step, corrected_step_count - 1)
        multiplier = self.schedule[idx] * .1
        mode = self.mode 
        if params.sigma.size(0) == 1 and mode != "both":
            mode = "both"
            if idx == 0:
                tqdm.write(f'\033[33mWARNING:\033[0m Forge does not support `cond` and `uncond` modes, using `both` instead')
        if mode == "cond":
            for i in range(self.batch_size):
                params.sigma[i] *= 1 - multiplier
        elif mode == "uncond":
            for i in range(self.batch_size):
                params.sigma[self.batch_size + i] *= 1 + multiplier
        else:
            params.sigma *= 1 - multiplier * self.cfg_scale
    
    def make_schedule(self, steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
        start = min(start, end)
        mid = start + bias * (end - start)
        multipliers = np.zeros(steps)

        start_idx, mid_idx, end_idx = [int(round(x * (steps - 1))) for x in [start, mid, end]]            

        start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
        if smooth:  
            start_values = 0.5 * (1 - np.cos(start_values * np.pi))
        start_values = start_values ** exponent
        if start_values.any():
            start_values *= (amount - start_offset)  
            start_values += start_offset  

        end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
        if smooth:
            end_values = 0.5 * (1 - np.cos(end_values * np.pi))
        end_values = end_values ** exponent
        if end_values.any():
            end_values *= (amount - end_offset)  
            end_values += end_offset  

        multipliers[start_idx:mid_idx+1] = start_values
        multipliers[mid_idx:end_idx+1] = end_values        
        multipliers[:start_idx] = start_offset
        multipliers[end_idx+1:] = end_offset    
        multipliers *= 1 - fade

        return multipliers

    def visualize(self, enabled, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
        try:
            steps = 50
            values = self.make_schedule(steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth)
            mean = sum(values)/steps
            peak = np.clip(max(abs(values)), -1, 1)
            if start > end:
                start = end
            mid = start + bias * (end - start)
            opacity = .1 + (1 - fade) * 0.7
            plot_color = (0.5, 0.5, 0.5, opacity) if not enabled else ((1 - peak)**2, 1, 0, opacity) if mean >= 0 else (1, (1 - peak)**2, 0, opacity) 
            plt.rcParams.update({
                "text.color":  plot_color, 
                "axes.labelcolor":  plot_color, 
                "axes.edgecolor":  plot_color, 
                "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),  
                "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),  
                "ytick.labelsize": 6,
                "ytick.labelcolor": plot_color,
                "ytick.color": plot_color,
            })            
            fig, ax = plt.subplots(figsize=(2.15, 2.00), layout="constrained")
            ax.plot(range(steps), values, color=plot_color)
            ax.axhline(y=0, color=plot_color, linestyle='dotted')
            ax.axvline(x=mid * (steps - 1), color=plot_color, linestyle='dotted')
            ax.tick_params(right=False, color=plot_color)
            ax.set_xticks([i * (steps - 1) / 10 for i in range(10)][1:])
            ax.set_xticklabels([])
            ax.set_ylim([-1, 1])
            ax.set_xlim([0, steps-1])
            plt.close()
            self.last_vis = fig
            return fig
        except Exception:
            if self.last_vis is not None:
                return self.last_vis
            return   


def xyz_support():
    for scriptDataTuple in scripts.scripts_data:
        if os.path.basename(scriptDataTuple.path) == 'xyz_grid.py':
            xy_grid = scriptDataTuple.module

            def confirm_mode(p, xs):
                for x in xs:
                    if x not in ['both', 'cond', 'uncond']:
                        raise RuntimeError(f'Invalid Detail Daemon Mode: {x}')
            mode = xy_grid.AxisOption(
                '[Detail Daemon] Mode',
                str,
                xy_grid.apply_field('DD_mode'),
                confirm=confirm_mode
            )
            amount = xy_grid.AxisOption(
                '[Detail Daemon] Amount',
                float,
                xy_grid.apply_field('DD_amount')
            )
            start = xy_grid.AxisOption(
                '[Detail Daemon] Start',
                float,
                xy_grid.apply_field('DD_start')
            )
            end = xy_grid.AxisOption(
                '[Detail Daemon] End',
                float,
                xy_grid.apply_field('DD_end')
            )
            bias = xy_grid.AxisOption(
                '[Detail Daemon] Bias',
                float,
                xy_grid.apply_field('DD_bias')
            )
            exponent = xy_grid.AxisOption(
                '[Detail Daemon] Exponent',
                float,
                xy_grid.apply_field('DD_exponent')
            )
            start_offset = xy_grid.AxisOption(
                '[Detail Daemon] Start Offset',
                float,
                xy_grid.apply_field('DD_start_offset')
            )
            end_offset = xy_grid.AxisOption(
                '[Detail Daemon] End Offset',
                float,
                xy_grid.apply_field('DD_end_offset')
            )
            fade = xy_grid.AxisOption(
                '[Detail Daemon] Fade',
                float,
                xy_grid.apply_field('DD_fade')
            )                                      
            xy_grid.axis_options.extend([
                mode,
                amount,
                start, 
                end, 
                bias, 
                exponent,
                start_offset,
                end_offset,
                fade,
            ])


try:
    xyz_support()
except Exception as e:
    print(f'Error trying to add XYZ plot options for Detail Daemon', e)
