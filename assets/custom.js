if (!window.dash_clientside) {
    window.dash_clientside = {};
}
window.dash_clientside.clientside = {
  make_draggable: function(id) {
    setTimeout(function() {
      var container = document.getElementById(id);
      var cards_final = []
      var cards = document.getElementsByClassName("card")

      
      var options = {
        accepts: function(el, target, source, sibling) {
          // Solo movemos dentro del mismo contenedor
          return target.parentElement==source.parentElement;
        },
        direction: 'vertical'
      };
      
      const containers = [container,...cards];
      var drake = dragula(containers, options);
      drake.on('drag', function(el, source) {
        options.direction="vertical"
      });
    }, 1)
    return window.dash_clientside.update
  },
  order: function (n_clicks){
    var pen = document.getElementsByClassName('theme');
    var string = "";
    for (var item of pen){
        string += item.id + ",";
    }
    console.log(string);
    return string;
  }
}

