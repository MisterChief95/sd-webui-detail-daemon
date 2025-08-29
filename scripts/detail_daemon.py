import os
import gradio as gr
import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import modules.scripts as scripts
from modules.script_callbacks import on_cfg_denoiser, remove_callbacks_for_function
from modules.shared import opts
from modules.ui_components import InputAccordion
from modules.infotext_utils import PasteField


allow_mode_select: bool

try:
    import modules_forge # noqa: F401
    allow_mode_select = False
except ImportError:
    allow_mode_select = True


class Script(scripts.Script):

    def __init__(self):
        super().__init__()
        self.schedule_params: dict[str, float] = None
        self.hr_schedule_params: dict[str, float] = None
        
        self.schedule = None
        
        self.is_hires = False
        self.is_hires_enabled = False
        self.callback_added = False
        self.is_img2img_fix_steps_changed = False
        self.img2img_fix_steps_old = None
        self.last_vis = None

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
                    preview = self.visualize(False, 0.2, 0.8, 0.5, 0.1, 1, 0, 0, 0, True, False, 0.2, 0.8, 0.5, 0.1, 1, 0, 0, 0, True)
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
                        gr_mode = gr.Dropdown(["both", "cond", "uncond"], value=lambda: "uncond" if allow_mode_select else "both", label="Mode", show_label=True, min_width=60, elem_classes=['detail-daemon-mode'], visible=allow_mode_select) 
                        gr_smooth = gr.Checkbox(label="Smooth", value=True, min_width=60, elem_classes=['detail-daemon-smooth'])

            gr_amount_slider.release(None, gr_amount_slider, gr_amount, _js="(x) => x")
            gr_amount.change(None, gr_amount, gr_amount_slider, _js="(x) => x")

            gr_start_offset_slider.release(None, gr_start_offset_slider, gr_start_offset, _js="(x) => x")
            gr_start_offset.change(None, gr_start_offset, gr_start_offset_slider, _js="(x) => x")

            gr_end_offset_slider.release(None, gr_end_offset_slider, gr_end_offset, _js="(x) => x")
            gr_end_offset.change(None, gr_end_offset, gr_end_offset_slider, _js="(x) => x")

            with InputAccordion(False, label="HiRes Fix", elem_id=self.elem_id('hr-detail-daemon'), visible=not is_img2img) as gr_hr_enabled:
                with gr.Row():
                    with gr.Column(scale=2):                    
                        gr_hr_amount_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.10, label="Detail Amount")
                        gr_hr_start = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.2, label="Start")
                        gr_hr_end = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.8, label="End") 
                        gr_hr_bias = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.5, label="Bias")                                                                                                                          
                with gr.Accordion("More Knobs:", elem_classes=['detail-daemon-more-accordion'], open=False):
                    with gr.Row():
                        with gr.Column(scale=2):   
                            with gr.Row():                                              
                                gr_hr_start_offset_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.00, label="Start Offset", min_width=60) 
                                gr_hr_end_offset_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.00, label="End Offset", min_width=60) 
                            with gr.Row():
                                gr_hr_exponent = gr.Slider(minimum=0.0, maximum=10.0, step=.05, value=1.0, label="Exponent", min_width=60) 
                                gr_hr_fade = gr.Slider(minimum=0.0, maximum=1.0, step=.05, value=0.0, label="Fade", min_width=60) 
                            # Because the slider max and min are sometimes too limiting:
                            with gr.Row():
                                gr_hr_amount = gr.Number(value=0.10, precision=4, step=.01, label="Amount", min_width=60)  
                                gr_hr_start_offset = gr.Number(value=0.0, precision=4, step=.01, label="Start Offset", min_width=60)  
                                gr_hr_end_offset = gr.Number(value=0.0, precision=4, step=.01, label="End Offset", min_width=60) 
                        with gr.Column(scale=1, min_width=275): 
                            gr_hr_mode = gr.Dropdown(["both", "cond", "uncond"], value=lambda: "uncond" if allow_mode_select else "both", label="Mode", show_label=True, min_width=60, elem_classes=['detail-daemon-mode'], visible=allow_mode_select) 
                            gr_hr_smooth = gr.Checkbox(label="Smooth", value=True, min_width=60, elem_classes=['detail-daemon-smooth'])

                gr_hr_amount_slider.release(None, gr_hr_amount_slider, gr_hr_amount, _js="(x) => x")
                gr_hr_amount.change(None, gr_hr_amount, gr_hr_amount_slider, _js="(x) => x")

                gr_hr_start_offset_slider.release(None, gr_hr_start_offset_slider, gr_hr_start_offset, _js="(x) => x")
                gr_hr_start_offset.change(None, gr_hr_start_offset, gr_hr_start_offset_slider, _js="(x) => x")

                gr_hr_end_offset_slider.release(None, gr_hr_end_offset_slider, gr_hr_end_offset, _js="(x) => x")
                gr_hr_end_offset.change(None, gr_hr_end_offset, gr_hr_end_offset_slider, _js="(x) => x")

            gr.Markdown("## [Ⓗ Help](https://github.com/muerrilla/sd-webui-detail-daemon)", elem_classes=['detail-daemon-help'])        

        controls = [
            gr_enabled, gr_mode, gr_start, gr_end, gr_bias, gr_amount, gr_exponent, gr_start_offset, gr_end_offset, gr_fade, gr_smooth,
            gr_hr_enabled, gr_hr_mode, gr_hr_start, gr_hr_end, gr_hr_bias, gr_hr_amount, gr_hr_exponent, gr_hr_start_offset, gr_hr_end_offset, gr_hr_fade, gr_hr_smooth,
        ]

        vis_args = controls.copy()
        vis_args.remove(gr_mode)
        vis_args.remove(gr_hr_mode)

        for vis_arg in vis_args:
            if isinstance(vis_arg, gr.components.Slider):
                vis_arg.release(fn=self.visualize, show_progress=False, inputs=vis_args, outputs=[gr_vis])
            else:
                vis_arg.change(fn=self.visualize, show_progress=False, inputs=vis_args, outputs=[gr_vis])

        self.infotext_fields = [
            PasteField(gr_enabled, lambda d: 'DD' in d or 'dd_' in d, api='dd_enabled'),
            PasteField(gr_mode, 'DD Mode', api='dd_mode'),
            PasteField(gr_amount, 'DD Amount', api='dd_amount'),
            PasteField(gr_start, 'DD Start', api='dd_start'),
            PasteField(gr_end, 'DD End', api='dd_end'),
            PasteField(gr_bias, 'DD Bias', api='dd_bias'),
            PasteField(gr_exponent, 'DD Exponent', api='dd_exponent'),
            PasteField(gr_start_offset, 'DD Start Offset', api='dd_start_offset'),
            PasteField(gr_end_offset, 'DD End Offset', api='dd_end_offset'),
            PasteField(gr_fade, 'DD Fade', api='dd_fade'),
            PasteField(gr_smooth, lambda d: 'DD Smooth' in d, api='dd_smooth'),
            PasteField(gr_hr_enabled, lambda d: 'DD HR' in d or 'dd_hr_' in d, api='dd_hr_enabled'),
            PasteField(gr_hr_mode, 'DD HR Mode', api='dd_hr_mode'),
            PasteField(gr_hr_amount, 'DD HR Amount', api='dd_hr_amount'),
            PasteField(gr_hr_start, 'DD HR Start', api='dd_hr_start'),
            PasteField(gr_hr_end, 'DD HR End', api='dd_hr_end'),
            PasteField(gr_hr_bias, 'DD HR Bias', api='dd_hr_bias'),
            PasteField(gr_hr_exponent, 'DD HR Exponent', api='dd_hr_exponent'),
            PasteField(gr_hr_start_offset, 'DD HR Start Offset', api='dd_hr_start_offset'),
            PasteField(gr_hr_end_offset, 'DD HR End Offset', api='dd_hr_end_offset'),
            PasteField(gr_hr_fade, 'DD HR Fade', api='dd_hr_fade'),
            PasteField(gr_hr_smooth, lambda d: 'DD HR Smooth' in d, api='dd_hr_smooth'),
        ]

        self.paste_field_names = []
        for field in self.infotext_fields:
            self.paste_field_names.append(field.api)

        return controls
    
    def process(self, p, 
                enabled, mode, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth,
                hr_enabled, hr_mode, hr_start, hr_end, hr_bias, hr_amount, hr_exponent, hr_start_offset, hr_end_offset, hr_fade, hr_smooth):

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
                "smooth": smooth,
            }

            if hr_enabled:
                self.hr_schedule_params = {
                    "start": hr_start,
                    "end": hr_end,
                    "bias": hr_bias,
                    "amount": hr_amount,
                    "exponent": hr_exponent,
                    "start_offset": hr_start_offset,
                    "end_offset": hr_end_offset,
                    "fade": hr_fade,
                    "smooth": hr_smooth,
                }

            self.mode = mode
            self.cfg_scale = p.cfg_scale
            self.batch_size = p.batch_size
            on_cfg_denoiser(self.denoiser_callback)              
            self.callback_added = True 
            
            p.extra_generation_params.update({
                "DD Mode": mode,
                "DD Amount": amount,
                "DD Start": start,
                "DD End": end,
                "DD Bias": bias,
                "DD Exponent": exponent,
                "DD Start Offset": start_offset,
                "DD End Offset": end_offset,
                "DD Fade": fade,
                "DD Smooth": smooth,
            })

            if hr_enabled:
                p.extra_generation_params.update({
                    "DD HR Mode": hr_mode,
                    "DD HR Amount": hr_amount,
                    "DD HR Start": hr_start,
                    "DD HR End": hr_end,
                    "DD HR Bias": hr_bias,
                    "DD HR Exponent": hr_exponent,
                    "DD HR Start Offset": hr_start_offset,
                    "DD HR End Offset": hr_end_offset,
                    "DD HR Fade": hr_fade,
                    "DD HR Smooth": hr_smooth,
                })
            
            tqdm.write('\033[32mINFO:\033[0m Detail Daemon is enabled')

        elif self.callback_added:
            remove_callbacks_for_function(self.denoiser_callback)
            self.callback_added = False
            self.schedule = None
            self.hr_schedule = None
            self.is_hires = self.is_hires_enabled = False


    def postprocess(self, p, processed, *args):
        if self.callback_added:
            remove_callbacks_for_function(self.denoiser_callback)
            self.callback_added = False
            self.schedule = None
            self.hr_schedule = None
            self.is_hires = self.is_hires_enabled = False
            opts.img2img_fix_steps = self.img2img_fix_steps_old if self.is_img2img_fix_steps_changed else opts.img2img_fix_steps

    def before_hr(self, p, 
            enabled, mode, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth,
            hr_enabled, hr_mode, hr_start, hr_end, hr_bias, hr_amount, hr_exponent, hr_start_offset, hr_end_offset, hr_fade, hr_smooth):
        
        self.is_hires = p.is_hr_pass or (hasattr(p, "txt2img_upscale") and p.txt2img_upscale)
        self.is_hires_enabled = self.is_hires and hr_enabled
        
        if not self.is_hires_enabled:
            return
        
        self.cfg_scale = p.hr_cfg
        self.mode = hr_mode
        self.schedule = None
        self.hr_schedule = None
        self.img2img_fix_steps_old = opts.img2img_fix_steps
        opts.img2img_fix_steps = True

        tqdm.write('\033[32mINFO:\033[0m Detail Daemon is enabled for Hires Fix')

    def setup(self, p, 
              enabled, mode, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth,
              hr_enabled, hr_mode, hr_start, hr_end, hr_bias, hr_amount, hr_exponent, hr_start_offset, hr_end_offset, hr_fade, hr_smooth):
        
        # Define parameter mappings
        base_params = ['mode', 'start', 'end', 'bias', 'amount', 'exponent', 'start_offset', 'end_offset', 'fade', 'smooth']
        hr_params = [f'hr_{param}' for param in base_params]
        
        # Set base parameters
        for param in base_params:
            setattr(p, param, locals()[param] if enabled else None)
        
        # Set HR parameters
        for param in hr_params:
            setattr(p, param, locals()[param] if enabled and hr_enabled else None)
        
        
    def denoiser_callback(self, params): 
        if self.is_hires and not self.is_hires_enabled:
            return
        
        total_steps = max(params.total_sampling_steps, params.denoiser.total_steps)
        corrected_step_count = total_steps - max(total_steps // params.denoiser.steps - 1, 0)

        if self.schedule is None:
            self.schedule = self.make_schedule(corrected_step_count, **(self.schedule_params if not self.is_hires else self.hr_schedule_params))

        step = max(params.sampling_step, params.denoiser.step)
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

    def visualize(self, 
                  enabled, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth,
                  hr_enabled, hr_start, hr_end, hr_bias, hr_amount, hr_exponent, hr_start_offset, hr_end_offset, hr_fade, hr_smooth):
        try:
            steps = 50
            base_values = self.make_schedule(steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth)
            mean = sum(base_values)/steps
            peak = np.clip(max(abs(base_values)), -1, 1)
            
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
            ax.plot(range(steps), base_values, color=plot_color, label="Base Schedule")
            ax.axhline(y=0, color=plot_color, linestyle='dotted')
            ax.axvline(x=mid * (steps - 1), color=plot_color, linestyle='dotted')
            
            if hr_enabled:
                hires_values = self.make_schedule(steps, hr_start, hr_end, hr_bias, hr_amount, hr_exponent, hr_start_offset, hr_end_offset, hr_fade, hr_smooth)
                hires_peak = np.clip(max(abs(hires_values)), -1, 1)
                hires_mean = sum(hires_values)/steps
                if hr_start > hr_end:
                    hr_start = hr_end
                hires_plot_color = (0.5, 0.5, 0.5, opacity) if not enabled else (1, (1 - hires_peak)**2, 1, opacity) if hires_mean >= 0 else (1, (1 - hires_peak)**2, 0, opacity)
                ax.plot(range(steps), hires_values, color=hires_plot_color, linestyle='-', label="Hires Schedule")
            
            ax.tick_params(right=False, color=plot_color)
            ax.set_xticks([i * (steps - 1) / 10 for i in range(10)][1:])
            ax.set_xticklabels([])
            ax.set_ylim((-1, 1))
            ax.set_xlim((0, steps-1))

            plt.close()

            self.last_vis = fig

            return fig
        
        except Exception as e:
            tqdm.write(f'\033[31mERROR:\033[0m Failed to visualize Detail Daemon schedule. Reason: {str(e)}')
            return self.last_vis if self.last_vis else None
        

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
                xy_grid.apply_field('dd_mode'),
                confirm=confirm_mode
            )
            amount = xy_grid.AxisOption(
                '[Detail Daemon] Amount',
                float,
                xy_grid.apply_field('dd_amount')
            )
            start = xy_grid.AxisOption(
                '[Detail Daemon] Start',
                float,
                xy_grid.apply_field('dd_start')
            )
            end = xy_grid.AxisOption(
                '[Detail Daemon] End',
                float,
                xy_grid.apply_field('dd_end')
            )
            bias = xy_grid.AxisOption(
                '[Detail Daemon] Bias',
                float,
                xy_grid.apply_field('dd_bias')
            )
            exponent = xy_grid.AxisOption(
                '[Detail Daemon] Exponent',
                float,
                xy_grid.apply_field('dd_exponent')
            )
            start_offset = xy_grid.AxisOption(
                '[Detail Daemon] Start Offset',
                float,
                xy_grid.apply_field('dd_start_offset')
            )
            end_offset = xy_grid.AxisOption(
                '[Detail Daemon] End Offset',
                float,
                xy_grid.apply_field('dd_end_offset')
            )
            fade = xy_grid.AxisOption(
                '[Detail Daemon] Fade',
                float,
                xy_grid.apply_field('dd_fade')
            )
            hr_mode = xy_grid.AxisOption(
                '[Detail Daemon] HR Mode',
                str,
                xy_grid.apply_field('dd_hr_mode'),
                confirm=confirm_mode
            )
            hr_amount = xy_grid.AxisOption(
                '[Detail Daemon] HR Amount',
                float,
                xy_grid.apply_field('dd_hr_amount')
            )
            hr_start = xy_grid.AxisOption(
                '[Detail Daemon] HR Start',
                float,
                xy_grid.apply_field('dd_hr_start')
            )
            hr_end = xy_grid.AxisOption(
                '[Detail Daemon] HR End',
                float,
                xy_grid.apply_field('dd_hr_end')
            )
            hr_bias = xy_grid.AxisOption(
                '[Detail Daemon] HR Bias',
                float,
                xy_grid.apply_field('dd_hr_bias')
            )
            hr_exponent = xy_grid.AxisOption(
                '[Detail Daemon] HR Exponent',
                float,
                xy_grid.apply_field('dd_hr_exponent')
            )
            hr_start_offset = xy_grid.AxisOption(
                '[Detail Daemon] HR Start Offset',
                float,
                xy_grid.apply_field('dd_hr_start_offset')
            )
            hr_end_offset = xy_grid.AxisOption(
                '[Detail Daemon] HR End Offset',
                float,
                xy_grid.apply_field('dd_hr_end_offset')
            )
            hr_fade = xy_grid.AxisOption(
                '[Detail Daemon] HR Fade',
                float,
                xy_grid.apply_field('dd_hr_fade')
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
                hr_mode,
                hr_amount,
                hr_start,
                hr_end,
                hr_bias,
                hr_exponent,
                hr_start_offset,
                hr_end_offset,
                hr_fade,
            ])


try:
    xyz_support()
except Exception as e:
    print(f'Error trying to add XYZ plot options for Detail Daemon', e)
