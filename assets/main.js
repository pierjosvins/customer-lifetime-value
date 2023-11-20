/*$('.cell-table tbody').on('dblclick', 'tr', function () {
console.log("Yes");
var data = table.row( this ).data();
alert( 'You clicked on '+data[0]+'\'s row' );
});

html.Button("Show details", className="btn btn-sm btn-info mb-1",id="display-details-modal")
console.log($('.cell-table tbody').val());


@app.callback(
    [Output('pred-value', 'value', allow_duplicate=True), Output('prob', 'value', )],
    [Input('open-details', 'n_clicks'), Input('cluster-graph', 'relayoutData')],
    prevent_initial_call=True
)
def no_update_when_opening_modal(n_clicks, relayoutData):

    if relayoutData != None:
       isAuto = False
       for key_ in list(relayoutData.keys()):
           if 'auto' in key_:
              isAuto = True
              break

    if(relayoutData != None and isAuto == False):
       x_values = [relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
       y_values = [relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']]
    else:
       x_values = [0., pred_data.PredictedCLV.max()]
       y_values = [0., pred_data.ProbBuy.max()]

    return x_values, y_values


*/
