# -*- coding: utf-8 -*-

from ipywidgets import (Box,
                        Button,
                        Checkbox,
                        Dropdown,
                        FlexBox,
                        FloatSlider,
                        FloatText,
                        HBox,
                        HTML,
                        Image,
                        IntSlider,
                        IntText,
                        RadioButtons,
                        SelectMultiple,
                        Tab,
                        Text,
                        ToggleButtons,
                        VBox,)
import ipywidgets as widgets
from IPython import display
from traitlets import HasTraits, link, Int, Float, Unicode, List, Bool
from elements import ELEMENTS
import getmolmap

LAYOUT_HTML_1 = '<style> \
.widget-area .getMolMap .panel-body{padding: 0;} \
.widget-area .getMolMap .widget-numeric-text{width: 2.5em;} \
.widget-area .getMolMap .widget-box.start{margin-left: 0;} \
.widget-area .getMolMap .widget-hslider{width: 20em;} \
.widget-area .getMolMap .widget-text{width: 10em;} \
</style>'


class SimpleDataModel(HasTraits):
    atomtype = Unicode('Pt')
    fold = Unicode('../demo_molecules')
    sub = Int(6)
    rad_type = Unicode('covrad')
    rad_scale = Float(1.17)
    radius = Float(0)
    radii = List(trait=Float, default_value=[0., ], minlen=1)
    excludeH = Bool(False)
    excludes = List(trait=Unicode)
    num_angles = Int(1)
    output_folder = Unicode('../results')
    output_name = Unicode('getmolmap_results')

    def get_values(self):
        values = dict([(k, getattr(self, k)) for k in self.trait_names()])
        values['radii'] = set([values['radius'], ] + values['radii'])
        #TODO: not elegant:
        dont = "Don't exclude any elements"
        values['excludes'] = [e for e in values['excludes'] if dont != e]
        if values['excludeH']:
            values['excludes'] = list(set(values['excludes'] + ['H']))
        return values


class PanelTitle(HTML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dom_classes += ("panel-heading panel-title",)


class PanelBody(Box):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dom_classes += ("panel-body",)


class ControlPanel(Box):
    # A set of related controls, with an optional title, in a box (provided by CSS)
    def __init__(self, title=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dom_classes += ("panel panel-info",)

        # add an option title widget
        if title is not None:
            self.children = [PanelTitle(value=title, margin=0, padding=6, border_radius=2,
                                        border_width=2),
                             PanelBody(children=self.children)]


class Alert(HTML):
    """HTML widget that is used to store alerts.  For now,
    just a pure HTML class but put in in separate class to
    allow potential customaization in future."""


class SimpleGui(Box):
    """
    An example GUI for a getMolMap application.
    Note that `self.model` is the owner of all of the "real" data, while this
    class handles creating all of the GUI controls and links. This ensures
    that the model itself remains embeddable and rem
    """

    def __init__(self, model=SimpleDataModel(), model_config=None, *args, **kwargs):
        self.model = model
        # Create alert widget (refactor into its own function?)
        alert = Alert(description = "Alert or something")
        # link((self.model, "message"), (alert, "value"))

        # Create a GUI
        # kwargs["orientation"] = 'vertical'
        kwargs["children"] = [self.INOUT_panel(), self.settings_panel()]
        #                    VBox([self.plot_panel(), self.slicing_panel(), self.unit_panel()]),])]

        super().__init__(*args, **kwargs)
        self._dom_classes += ("getMolMap row",)


    def tight_layout(self):
        """ Tight layout for gui boxes/widgets """
        return display.HTML(LAYOUT_HTML_1)

    def INOUT_panel(self):
        # create correlation controls. NOTE: should only be called once.
        # loadbutton = Button(color='black', background_color='AliceBlue',
        #                     description="Upload Geometry", margin=0, padding=3)
        file_widget = FileWidget()
        self.model.file_widget = file_widget  # This line is for debugging only.

        # Register an event to echo the filename when it has been changed.
        def file_loading():
            print('Loading "{}" ...'.format(file_widget.filename), end='', flush=True)
        file_widget.on_trait_change(file_loading, 'filename')

        # Register an event to echo the filename and contents when a file
        # has been uploaded.
        def file_loaded():
            print(" done.")
            with open('../moldata/{}'.format(file_widget.filename), 'w') as f:
                f.write(file_widget.value)
        file_widget.on_trait_change(file_loaded, 'value')

        # Register an event to print an error message when a file could not
        # be opened.  Since the error messages are not handled through
        # traitlets but instead handled through custom msgs, the registration
        # of the handler is different than file_loading and file_loaded above.
        # Instead the API provided by the CallbackDispatcher must be used.
        def file_failed():
            print("Could not load file contents of %s" % file_widget.filename)
        file_widget.errors.register_callback(file_failed)

        # Downloading results is not working yet. Instruct the user to use the dashboard.
        #TODO: show message to use the dashboard
        #TODO: implement download properly
        savebutton = Button(color='#586e75', background_color='#eee8d5',
                            description="Export Results", margin=0, padding=1)
        self.model.savebutton = savebutton
        button_gap = Box(margin=11, background_color='blue')
        area = HBox(children=[file_widget, button_gap, savebutton], margin=0)
        return ControlPanel(title="Upload geometry:", children=[area],
                            border_width=2, border_radius=4, margin=0, padding=0)

    def settings_panel(self):
        # getMolMap calulation settings.  NOTE: should only be called once.
        margin = 2

        sub_slider_text = "Subdivision value of the icosphere for numerical calculation:"
        sub_slider_widget = IntSlider(value=5, min=1, max=8,)
        link((self.model, 'sub'), (sub_slider_widget, 'value'))
        sub_slider = VBox(children=[HTML(value=sub_slider_text), sub_slider_widget],
                          margin=margin, width='100%')
#        link((sub_slider, 'value'), (i, 'value'))
#        print(self.width)
#        sub_slider.align = 'center'
#        sub_slider.width = '100%'
#        sub_slider.border_color = 'black'
#        sub_slider.border_width = 2

        radius_slider_text = "Cut radius measured from the central atom:"
        radius_slider_widget = FloatSlider(value=0, min=0, max=10)
        link((self.model, 'radius'), (radius_slider_widget, 'value'))
        radius_slider = VBox(children=[HTML(value=radius_slider_text), radius_slider_widget],
                             margin=margin,)

        atomradscale_slider_text = "Atomic radius scaling factor:"
        atomradscale_slider_widget = FloatSlider(value=1, min=0, max=4)
        link((self.model, 'rad_scale'), (atomradscale_slider_widget, 'value'))
        atomradscale_slider = VBox(children=[HTML(value=atomradscale_slider_text),
                                             atomradscale_slider_widget],
                                   margin=margin,)

        excludeH_button = Checkbox(description='Exclude H',)
        link((self.model, 'excludeH'), (excludeH_button, 'value'))

        dont = "Don't exclude any elements"
        #TODO: Syncronize exclude_list_widget with excludeH button and define an event on the
        # model to filter out the `dont` text.
        # Alternatevily, separate this option into a checkbox and hide the exclude options
        # while the button is selected.
        exclude_list_text = 'Exclude:'
        exclude_list_widget = SelectMultiple(options=[dont] + [e.symbol for e in ELEMENTS],
                                             selected_labels=[dont],
                                             color='Black',
                                             font_size=14,
                                             height=120)
        link((exclude_list_widget, 'value'), (self.model, 'excludes'))
        exclude_list = VBox(children=[HTML(value=exclude_list_text), exclude_list_widget],
                            margin=margin)

        atomrad_button = ToggleButtons(description='Atomic radius type:',
                                       options=['vdwrad', 'covrad', 'atmrad'],
                                       background_color='AliceBlue',
                                       margin=margin)
        link((self.model, 'rad_type'), (atomrad_button, 'value'))
        runbutton = Button(description="Run calculation!",
                           tooltip='Click here to calculate coverage and inverse cone angles!',
                           margin=margin * 3,
                           border_color='#9acfea',
                           # border_radius=5,
                           border_width=3,
                           font_size=20)
        runbutton.on_click(self.run_button_clicked)

        basic_tab = VBox(children=[atomrad_button, excludeH_button, ])
        sliders = VBox(children=[atomradscale_slider, radius_slider, sub_slider])
        sliders.width = '100%'
        sliders.pack = 'center'

        advanced_tab = VBox(children=[atomrad_button, sliders, exclude_list])
        main_window = Tab(children=[basic_tab, advanced_tab])
        main_window.set_title(0, 'Basic')
        main_window.set_title(1, 'Advanced')

        return ControlPanel(title="getMolMap Settings", children=[main_window, runbutton],
                            border_width=2, border_radius=4, margin=10, padding=0)

    def run_button_clicked(self, trait_name):
        kwargs = self.model.get_values()
        getmolmap.calc(**kwargs)

    def output_panel(self):
        pass


class FileWidget(widgets.DOMWidget):
    _view_name = Unicode('FilePickerView', sync=True)
    value = Unicode(sync=True)
    filename = Unicode(sync=True)

    def __init__(self, **kwargs):
        """Constructor"""
        widgets.DOMWidget.__init__(self, **kwargs) # Call the base.

        # Allow the user to register error callbacks with the following signatures:
        #    callback()
        #    callback(sender)
        self.errors = widgets.CallbackDispatcher(accepted_nargs=[0, 1])

        # Listen for custom msgs
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, content):
        """Handle a msg from the front-end.

        Parameters
        ----------
        content: dict
            Content of the msg."""
        if 'event' in content and content['event'] == 'error':
            self.errors()
            self.errors(self)