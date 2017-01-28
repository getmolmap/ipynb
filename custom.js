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

 requirejs.undef('filepicker');

 define('filepicker', ["jupyter-js-widgets"], function(widgets) {

     var FilePickerView = widgets.DOMWidgetView.extend({
         render: function(){
             // Render the view using HTML5 multiple file input support.
             this.setElement($('<input class="fileinput" multiple="multiple" name="datafile"  />')
                 .attr('type', 'file'));
         },

         events: {
             // List of events and their handlers.
             'change': 'handle_file_change',
         },

         handle_file_change: function(evt) {
             // Handle when the user has changed the file.

             // Save context (or namespace or whatever this is)
             var that = this;

             // Retrieve the FileList object
             var files = evt.originalEvent.target.files;
             var filenames = [];
             var file_readers = [];
             console.log("Reading files:");

             for (var i = 0; i < files.length; i++) {
                 var file = files[i];
                 console.log("Filename: " + file.name);
                 console.log("Type: " + file.type);
                 console.log("Size: " + file.size + " bytes");
                 filenames.push(file.name);

                 // Read the file's textual content and set value_i to those contents.
                 file_readers.push(new FileReader());
                 file_readers[i].onload = (function(file, i) {
                     return function(e) {
                         that.model.set('value_' + i, e.target.result);
                         that.touch();
                         console.log("file_" + i + " loaded: " + file.name);
                     };
                 })(file, i);

                 file_readers[i].readAsText(file);
             }

             // Set the filenames of the files.
             this.model.set('filenames', filenames);
             this.touch();
         },
     });

     // Register the FilePickerView with the widget manager.
     return {
         FilePickerView: FilePickerView
     };
 });

// define([
//         'base/js/namespace',
//         'base/js/events'
//     ],
//     function(IPython, events) {
//         events.on("app_initialized.NotebookApp",
//             function() {
//                 IPython.Cell.options_default.cm_config.lineNumbers = true;
//             }
//         );
//     }
// );
