
// Set up the plot window.
        var margin = 80;
        var w = 700 - 2 * margin, h = 500 - 2 * margin;
        var svg = d3.select("#plot").append("svg")
                        .attr("width", w + 2 * margin)
                        .attr("height", h + 2 * margin)
                    .append("svg:g")
                        .attr("transform", "translate(" + margin + ", " + margin + ")");

        // The colorbar.
        var color = d3.scale.quantize()
                      .range(["#156b87", "#876315", "#543510", "#872815"])
                      .domain([0, 1]);

        // Axes scaling functions.
        var xscale = d3.scale.linear().range([0, w]);
        var yscale = d3.scale.linear().range([h, 0]);

        // The axes objects themselves.
        var xaxis = d3.svg.axis().scale(xscale).ticks(8);
        var yaxis = d3.svg.axis().scale(yscale).ticks(8).orient("left");

        svg.append("svg:g").attr("class", "x axis")
                           .attr("transform", "translate(0, " + h + ")");
        svg.append("svg:g").attr("class", "y axis");

        // Show the information about a particular point.
        var show_info = function (d) {
            d3.select("#point-info").text("This is point " + d._id + ". "
                + "It has the coordinates (" + d.x + ", " + d.y + ").");
        };

        // Load the data.
        var callback = function (data) {
            // Rescale the axes.
            xscale.domain([d3.min(data, function (d) { return d.x; }) - 0.05,
                           d3.max(data, function (d) { return d.x; }) + 0.05]);
            yscale.domain([d3.min(data, function (d) { return d.y; }) - 0.05,
                           d3.max(data, function (d) { return d.y; }) + 0.05]);

            // Display the axes.
            svg.select(".x.axis").call(xaxis);
            svg.select(".y.axis").call(yaxis);

            // Insert the data points.
            svg.selectAll("circle").data(data).enter()
                .append("circle")
                    .attr("id", function (d) { return d._id; })
                    .attr("cx", function (d) { return xscale(d.x); })
                    .attr("cy", function (d) { return yscale(d.y); })
                    .attr("r", function (d) { return 2 * Math.sqrt(d.area); })
                    .style("fill", function (d) { return color(d.color); })
                    .on("mousedown", show_info);
        };

        d3.json("/data", callback);
