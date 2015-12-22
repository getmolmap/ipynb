// leave at least 2 line with only a star on it below, or doc generation fails
/**
 *
 *
 * Placeholder for custom user javascript
 * mainly to be overridden in profile/static/custom/custom.js
 * This will always be an empty file in IPython
 *
 * User could add any javascript in the `profile/static/custom/custom.js` file
 * (and should create it if it does not exist).
 * It will be executed by the ipython notebook at load time.
 *
 * Same thing with `profile/static/custom/custom.css` to inject custom css into the notebook.
 *
 * Example :
 *
 * Create a custom button in toolbar that execute `%qtconsole` in kernel
 * and hence open a qtconsole attached to the same kernel as the current notebook
 *
 *    $([IPython.events]).on('app_initialized.NotebookApp', function(){
 *        IPython.toolbar.add_buttons_group([
 *            {
 *                 'label'   : 'run qtconsole',
 *                 'icon'    : 'icon-terminal', // select your icon from http://fortawesome.github.io/Font-Awesome/icons
 *                 'callback': function () {
 *                     IPython.notebook.kernel.execute('%qtconsole')
 *                 }
 *            }
 *            // add more button here if needed.
 *            ]);
 *    });
 *
 * Example :
 *
 *  Use `jQuery.getScript(url [, success(script, textStatus, jqXHR)] );`
 *  to load custom script into the notebook.
 *
 *    // to load the metadata ui extension example.
 *    $.getScript('/static/notebook/js/celltoolbarpresets/example.js');
 *    // or
 *    // to load the metadata ui extension to control slideshow mode / reveal js for nbconvert
 *    $.getScript('/static/notebook/js/celltoolbarpresets/slideshow.js');
 *
 *
 * @module IPython
 * @namespace IPython
 * @class customjs
 * @static
 */

require(["widgets/js/widget", "widgets/js/manager"], function(widget, manager){

    var FilePickerView = widget.DOMWidgetView.extend({
        render: function(){
            // Render the view.
            this.setElement($('<input />')
                .attr('type', 'file'));
        },
        
        events: {
            // List of events and their handlers.
            'change': 'handle_file_change',
        },
       
        handle_file_change: function(evt) { 
            // Handle when the user has changed the file.
            
            // Retrieve the first (and only!) File from the FileList object
            var file = evt.target.files[0];
            if (file) {

                // Read the file's textual content and set value to those contents.
                var that = this;
                var file_reader = new FileReader();
                file_reader.onload = function(e) {
                    that.model.set('value', e.target.result);
                    that.touch();
                }
                file_reader.readAsText(file);
            } else {

                // The file couldn't be opened.  Send an error msg to the
                // back-end.
                this.send({ 'event': 'error' });
            }

            // Set the filename of the file.
            this.model.set('filename', file.name);
            this.touch();
        },
    });
        
    // Register the FilePickerView with the widget manager.
    manager.WidgetManager.register_widget_view('FilePickerView', FilePickerView);
});

$([IPython.events]).on('app_initialized.NotebookApp', function() {
    // Add the shortcut

    IPython.keyboard_manager.command_shortcuts.add_shortcut('ctrl-x', {
        help: 'Clear all output', // This text will show up on the
        handler: function(event) { //  help page (CTRL-M h or ESC h)
            IPython.notebook.clear_all_output(); // Function that gets invoked and
            return false; //  triggers a notebook command
        }
    });

    IPython.keyboard_manager.command_shortcuts.add_shortcut('ctrl-r', {

        help: 'Run all above including this cell', // This text will show up on the
        handler: function(event) { //  help page (CTRL-M h or ESC h)
            IPython.notebook.execute_cells_above();
            IPython.notebook.select_next();
            IPython.notebook.execute_cell();
            return false; //  triggers a notebook command
        }
    });

    IPython.keyboard_manager.edit_shortcuts.add_shortcut('ctrl-r', {

        help: 'Run all above including this cell', // This text will show up on the
        handler: function(event) { //  help page (CTRL-M h or ESC h)
            IPython.notebook.execute_cells_above();
            IPython.notebook.select_next();
            IPython.notebook.execute_cell();
            return false; //  triggers a notebook command
        }
    });

    IPython.keyboard_manager.edit_shortcuts.add_shortcut('cmd-enter', {

        help: 'Run current cell', // This text will show up on the
        handler: function(event) { //  help page (CTRL-M h or ESC h)
            IPython.notebook.execute_cell();
            return false; //  triggers a notebook command
        }
    });

    IPython.keyboard_manager.command_shortcuts.add_shortcut('cmd-enter', {

        help: 'Run current cell', // This text will show up on the
        handler: function(event) { //  help page (CTRL-M h or ESC h)
            IPython.notebook.execute_cell();
            return false; //  triggers a notebook command
        }
    });
    // A small hint so we can see through firebug that our custom code executed
    console.log("Customtcut(s) loaded");
});

define([
        'base/js/namespace',
        'base/js/events'
    ],
    function(IPython, events) {
        events.on("app_initialized.NotebookApp",
            function() {
                IPython.Cell.options_default.cm_config.lineNumbers = true;
            }
        );
    }
);
