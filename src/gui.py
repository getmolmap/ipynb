# -*- coding: utf-8 -*-

import os
from collections import OrderedDict
from time import sleep, time
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
                        VBox,
                        jslink, )
import ipywidgets as widgets
from IPython import display
from traitlets import Bool, Dict, Float, HasTraits, Int, link, List, Unicode
from elements import ELEMENTS
import getmolmap
from icosaio import getxyz

LAYOUT_HTML_1 = '<style> \
.widget-area .getMolMap .panel-body{padding: 0;} \
.widget-area .getMolMap .widget-numeric-text{width: 2.5em;} \
.widget-area .getMolMap .widget-box.start{margin-left: 0;} \
.widget-area .getMolMap .widget-hslider{width: 20em;} \
.widget-area .getMolMap .widget-text{width: 10em;} \
</style>'


class SimpleDataModel(HasTraits):
    atom_counts = Dict(sync=True)
    file_names = List(trait=Unicode, sync=True)  # default_value=['iprc.xyz', 'iprc2.xyz'],
    atom_types = List(trait=Unicode, sync=True)  # default_value=['Pt', 'Pt'],
    centrum_nums = List(trait=List(trait=Int), sync=True)  # default_value=[[5], [5]],
    fold = Unicode('./moldata', sync=True)
    sub = Int(6, sync=True)
    rad_type = Unicode('covrad', sync=True)
    rad_scale = Float(1.17, sync=True)
#    rad_scales = List(trait=Float, default_value=[1.17, 1.17], sync=True)
#    radii = List(trait=List(trait=Float), default_value=[[0.], [0.]], sync=True)
    radius = Float(0.0, sync=True)
    excludeH = Bool(False, sync=True)
    excludes = List(trait=Unicode, default_value=['H'], sync=True)
    num_angles = Int(1, sync=True)
    output_folder = Unicode('./results', sync=True)
    output_name = Unicode('getmolmap_results', sync=True)
    table = Bool(True, sync=True)
    advanced_tab_visible = Bool(False, sync=True)
    debug_level = Int(0, sync=True)

    def get_values(self):
        values = dict([(k, getattr(self, k)) for k in self.trait_names()])
        values['hdf5_path'] = values.get('hdf5_path', os.path.abspath('./progdata'))
        values['radii'] = [[0.] for i in values['atom_types']]
        values['rad_scales'] = [values['rad_scale'] for i in values['atom_types']]
        #TODO: not elegant:
        dont = "Don't exclude any elements"
        values['excludes'] = [e for e in values['excludes'] if e != dont]
        values['num_angles'] = [values['num_angles']] * len(values['atom_types'])
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
    download_link = Unicode(sync=True)
    results_table = Unicode(sync=True)

    def __init__(self, model=SimpleDataModel(), model_config=None, *args, **kwargs):
        self.model = model
        # Create alert widget (refactor into its own function?)
        alert = Alert(description = "Alert or something")
        # link((self.model, "message"), (alert, "value"))

        # Create a GUI
        # kwargs["orientation"] = 'vertical'
        kwargs["children"] = [self.INOUT_panel(), self.settings_panel(), self.output_panel()]
        #                    VBox([self.plot_panel(), self.slicing_panel(), self.unit_panel()]),])]

        super().__init__(*args, **kwargs)
        self._dom_classes += ("getMolMap row",)
        self.tight_layout()



    def tight_layout(self):
        """ Tight layout for gui boxes/widgets """
        return display.HTML(LAYOUT_HTML_1)

    def INOUT_panel(self):
        # create correlation controls. NOTE: should only be called once.
        # loadbutton = Button(color='black', background_color='AliceBlue',
        #                     description="Upload Geometry", margin=0, padding=3)
        upload_area = VBox()
        file_widget = FileWidget()
        self.model.file_widget = file_widget  # This line is for debugging only.

        def centrum_changed(name, old, new):
            '''When the user selects another centrum atom type, change the options for the
            atomnum_picker widget to the possible numbers'''
            i = int(name.strip('_'))
            print(self.model.file_names[i], 'old, new:', old, new, end=' ', flush=True)
            if new != old:
                i = int(name.strip('_'))
                fname = self.model.file_names[i]
                options = self.model.atom_counts[fname][new]
                atomnum_picker1 = upload_area.children[i].children[1]
                atomnum_picker1.options = OrderedDict([('-', 0)] + [(str(i), i) for i in options])
                self.model.atom_types[i] = new
                values = atomnum_picker1.options.values()
                value = [v for v in values][1]
                atomnum_picker1.value = value

        def atomnum_changed(name, old, new):
            if new != old:
                i = int(name.strip('_'))
                for j in range(1,4):
                    atomnum_pickerj = upload_area.children[i].children[j]
                    self.model.centrum_nums[i][j - 1] = int(atomnum_pickerj.value)

        def file_loaded(name, new):
            '''Register an event to save contents when a file has been uploaded.'''
            i = int(name.split('_')[1])
            fname = file_widget.filenames[i]
            index = self.model.file_names.index(fname)
            len_files = len(self.model.file_names)
            fpath = os.path.join('./moldata', fname)
            with open(fpath, 'w') as f:
                f.write(new)
            geom = getxyz(fpath)
            atom_count = {}
            heaviest = 1
            for i, line in enumerate(geom):
                anum = int(line[0])
                heaviest = max(anum, heaviest)
                symbol = ELEMENTS[anum].symbol
                if symbol in atom_count.keys():
                    atom_count[symbol].append(i + 1)
                else:
                    atom_count[symbol] = [i + 1]
            self.model.atom_counts[fname] = atom_count
            counter = 1000
            while counter:
                if len(upload_area.children) == len_files:
                    element_picker = upload_area.children[index].children[0]
                    element_picker.options = list(atom_count.keys())
                    element_picker.value = ELEMENTS[heaviest].symbol
                    return None
                else:
                    counter -= 1
                    sleep(0.005)
            print('fileLoadError:', fname, flush=True)

        # Register an event to echo the filename when it has been changed.
        def file_loading(name, old, new):
            '''Update self.model when user requests a list of files to be uploaded'''


            traits = [('value_{}'.format(i), Unicode(sync=True)) for i in range(len(new))]
            file_widget.add_traits(**dict(traits))
            for i in range(len(new)):
                file_widget.on_trait_change(file_loaded, 'value_{}'.format(i))

            # file_names are uniqe, we ignore any duplicates
            old_fnames = self.model.file_names
            file_names = [fn for fn in new if fn not in old_fnames]
            old_len = len(old_fnames)
            new_len = len(file_names)
            self.model.file_names.extend(file_names)
            self.model.atom_types.extend(['H' for i in range(new_len)])
            self.model.centrum_nums.extend([[1, 1, 1] for i in range(new_len)])
            old_children = list(upload_area.children)
            new_children = []
            options = {'-': 0, }
            for i, file_name in zip(range(old_len, new_len), file_names):
                j = '_{}'.format(i)
                element_picker = Dropdown(description=file_name, value='', options=[''],
                                          color='Black', height=32, width=32, font_size=14,)
                element_picker.add_traits(**{j: Unicode(sync=True)})
                link((element_picker, 'value'), (element_picker, j))
                element_picker.on_trait_change(centrum_changed, j)
                atomnum_picker1 = Dropdown(options=options, value=0, color='Black',
                                          height=32, font_size=14,)
                atomnum_picker2 = Dropdown(options=options, value=0, color='Black',
                                          height=32, font_size=14,)
                atomnum_picker3 = Dropdown(options=options, value=0, color='Black',
                                          height=32, font_size=14,)
                link((atomnum_picker1, 'options'), (atomnum_picker2, 'options'))
                link((atomnum_picker2, 'options'), (atomnum_picker3, 'options'))

                atomnum_picker1.add_traits(**{j: Int(sync=True)})
                atomnum_picker2.add_traits(**{j: Int(sync=True)})
                atomnum_picker3.add_traits(**{j: Int(sync=True)})
                link((atomnum_picker1, 'value'), (atomnum_picker1, j))
                link((atomnum_picker2, 'value'), (atomnum_picker2, j))
                link((atomnum_picker3, 'value'), (atomnum_picker3, j))
                atomnum_picker1.on_trait_change(atomnum_changed, j)
                atomnum_picker2.on_trait_change(atomnum_changed, j)
                atomnum_picker3.on_trait_change(atomnum_changed, j)

                line = HBox([element_picker, atomnum_picker1, atomnum_picker2, atomnum_picker3])
                new_children.append(line)
            upload_area.children = old_children + new_children
            # print('Loading {} file(s)...'.format(file_widget.num_files), end='', flush=True)
        file_widget.on_trait_change(file_loading, 'filenames')
#        self.model.file_widget = file_widget


#            file_widget.traits(name) = ''

        # Register an event to print an error message when a file could not
        # be opened.  Since the error messages are not handled through
        # traitlets but instead handled through custom msgs, the registration
        # of the handler is different than file_loading and file_loaded above.
        # Instead the API provided by the CallbackDispatcher must be used.
        def file_failed():
            print("Could not load some file contents.")
        file_widget.errors.register_callback(file_failed)

        button_gap = Box(margin=11, background_color='blue')
        button_area = HBox([file_widget, button_gap, ], margin=0)
        area = VBox([button_area, upload_area])
        formats = HTML(value='Supported file extensions: .xyz, .pdb, .cif, and plenty \
        <a href="http://openbabel.org/wiki/List_of_extensions" target="_blank"> more</a>.')
        return ControlPanel(title="Upload geometry:", children=[formats, area],
                            border_width=2, border_radius=4, margin=0, padding=0)

    def settings_panel(self):
        # getMolMap calulation settings.  NOTE: should only be called once.
        margin = 2

        num_angles_slider_text = "Number of Inverse Cone Angles to calculate:"
        num_angles_slider_widget = IntSlider(value=1, min=1, max=5,)
        num_angles_slider = VBox(children=[HTML(value=num_angles_slider_text),
                                           num_angles_slider_widget],
                                 margin=margin, width='100%')
        link((self.model, 'num_angles'), (num_angles_slider_widget, 'value'))
        sub_slider_text = "Subdivision value of the icosphere for numerical calculation:"
        sub_slider_widget = IntSlider(value=5, min=1, max=9,)
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

        excludeH_button = Checkbox(description='Exclude H from every geometry:',)
        link((self.model, 'excludeH'), (excludeH_button, 'value'))
        excludeH_button.on_trait_change(self.excludeH_changed, 'value')

        dont = "Don't exclude any elements"
        self.dont = dont
        #TODO: Syncronize exclude_list_widget with excludeH button and define an event on the
        # model to filter out the `dont` text.
        # Alternatevily, separate this option into a checkbox and hide the exclude options
        # while the button is selected.
        exclude_list_text = 'Exclude elements from every geometry:'
        exclude_list_widget = SelectMultiple(options=[dont] + [e.symbol for e in ELEMENTS],
                                             selected_labels=[dont],
                                             color='Black',
                                             font_size=14,
                                             height=120)
        link((exclude_list_widget, 'value'), (self.model, 'excludes'))
        # The dirty old SelectMultiple widget does not have an .on_trait_change method.
        # So we create a new traitlet (excludes_notifier), which has an .on_trait_change method
        # because it inherits HasTraits. We link the 'value' trait to excludes_notifier.excludes;
        # and we bind the event handler to excludes_notifier.on_trait_change
        self.excludes_notifier = ExcludesNotifier()
        link((exclude_list_widget, 'value'), (self.excludes_notifier, 'excludes'))
        self.excludes_notifier.on_trait_change(self.excludes_changed)

        exclude_list = VBox(children=[HTML(value=exclude_list_text), exclude_list_widget],
                            margin=margin)

        atomrad_button = ToggleButtons(description='Atomic radius type:',
                                       options=['vdwrad', 'covrad', 'atmrad'],
                                       background_color='AliceBlue',
                                       margin=margin)
        link((self.model, 'rad_type'), (atomrad_button, 'value'))
        runbutton = Button(description="Run calculation!",
                           tooltip=
                           'Click here to calculate Buried Volumes and Inverse Cone Angles!',
                           margin=margin * 3,
                           border_color='#9acfea',
                           # border_radius=5,
                           border_width=3,
                           font_size=20)
        runbutton.on_click(self.run_button_clicked)

        basic_tab = VBox(children=[atomrad_button, excludeH_button, ])
        sliders = VBox(children=[num_angles_slider, atomradscale_slider, radius_slider,
                                 sub_slider])
        sliders.width = '100%'
        sliders.pack = 'center'

        advanced_tab = VBox(children=[atomrad_button, sliders, exclude_list])
        main_window = Tab(children=[basic_tab, advanced_tab])
        main_window.set_title(0, 'Basic')
        main_window.set_title(1, 'Advanced')

        return ControlPanel(title="getMolMap Settings:", children=[main_window, runbutton],
                            border_width=2, border_radius=4, margin=10, padding=0)

    def excludeH_changed(self, *args, **kwargs):
        if self.model.excludeH:
            self.model.excludes = list(set(self.model.excludes) | {'H', })
            self.model.excludes = list(set(self.model.excludes) - {self.dont, })
        else:
            self.model.excludes = list(set(self.model.excludes) - {'H', })
            if len(self.model.excludes) > 1 and self.dont in self.model.excludes:
                self.model.excludes = list(set(self.model.excludes) - {self.dont, })

    def excludes_changed(self, *args, **kwargs):
        if 'H' in self.model.excludes and not self.model.excludeH:
            self.model.excludeH = True
        elif 'H' not in self.model.excludes and self.model.excludeH:
            self.model.excludeH = False

    def run_button_clicked(self, trait_name):
        kwargs = self.model.get_values()
        html_table = getmolmap.calc(**kwargs)
        self.download_link = 'Download <a href="/files/results/getmolmap_results.xlsx"\
         target="_blank"> getmolmap_results.xlsx</a>.'
        self.results_table = html_table


    def output_panel(self):
        download_link = HTML(value='')
        link((self, 'download_link'), (download_link, 'value'))
        results_table = HTML(value='')
        link((self, 'results_table'), (results_table, 'value'))
        content = VBox(children=[download_link, results_table])
        return ControlPanel(title="Results:", children=[content],
                            border_width=2, border_radius=4, margin=10, padding=0)
        """<div style="height:100px;width:200px;overflow:auto;border:8px solid yellowgreen;padding:2%">This </div>"""


class ExcludesNotifier(HasTraits):
    excludes = List(trait=Unicode, default_value=['H'], sync=True)


class FileWidget(widgets.DOMWidget):
    _view_name = Unicode('FilePickerView', sync=True)
#    value = Unicode(sync=True)
#    file_name = Unicode(sync=True)
    filenames = List([], sync=True)
#    values = List(trait=Unicode, sync=True)

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
