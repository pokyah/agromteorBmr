# build the ML task and do not keep sid as it is not a var
# devtools::install_github("pokyah/mlr", ref = "gstat")
# https://www.sciencedirect.com/science/article/pii/S2211675315000482
# https://stackoverflow.com/questions/40527442/r-mlr-wrapper-feature-selection-hyperparameter-tuning-without-nested-nested
# https://github.com/mlr-org/mlr/issues/1861
# https://mlr.mlr-org.com/articles/tutorial/handling_of_spatial_data.html
# https://www.youtube.com/watch?v=LpOsxBeggM0


# 2016-09-11 17:00:00

# load the required libraries
library(dplyr)
# devtools::install_github("r-spatial/sf")
library(sf)
# devtools::install_github("pokyah/agrometVars")
library(agrometeorVars)
# installing new version of agrometAPI
#devtools::install_github("pokyah/agrometAPI")
library(agrometAPI)
# installing custo mversion of mlr
# devtools::install_github("pokyah/mlr", ref="gstat")
library(mlr)
library(leaflet)
library(leaflet.extras)

# loading the datasets from agrometeorVars package
data("grid.sf")
data("grid.static")
data("grid.dyn")

data("stations.sf")
data("stations.static")
data("stations.dyn")

data("intersections")

# newdata on which to predict
newdata = grid.static
newdata = sf::st_transform(newdata, 3812)
coords = data.frame(sf::st_coordinates(newdata))
st_geometry(newdata) = NULL
newdata = newdata %>%
  dplyr::bind_cols(coords) %>%
  dplyr::rename(x = X) %>%
  dplyr::rename(y = Y)

# defining the function that create a dataset for a specific hour (mtime)
make.1h.data = function(dateTime){
  # extracting the tsa_hp1 var at stations points for current mtime
  stations.mtime = stations.dyn %>%
    # filtering for current time
    dplyr::filter(mtime == dateTime) %>%
    # adding corresponding grid px
    dplyr::left_join(
      data.frame(stations.sf)[c("sid", "px")],
      by = "sid"
    ) %>%
    # adding tsa_hp1
    dplyr::left_join(
      (grid.dyn %>%
          dplyr::filter(mtime == dateTime) %>%
          dplyr::select(c("px", "tsa_hp1"))
      ),
      by = "px"
    ) %>%
    # adding static vars
    dplyr::left_join(
      stations.static,
      by = "sid"
    ) %>%
    # removing useless px
    dplyr::select(-px) %>%
    # adding the lat and lon as projected Lambert 2008 (EPSG = 3812)
    dplyr::left_join(
      (data.frame(st_coordinates(st_transform(stations.sf, 3812))) %>%
          dplyr::bind_cols(stations.sf["sid"]) %>%
          dplyr::select(-geometry)
      ),
      by = "sid"
    ) %>%
    # removing mtime
    dplyr::select(-mtime) %>%
    # removing tsa_hp1 because missing values
    dplyr::select(-tsa_hp1) %>%
    # renaming X and Y to x and y
    dplyr::rename(x = X) %>%
    dplyr::rename(y = Y)
}
# defining the function that performs the benchmark
make.1h.bmr = function(task, dateTime, learners){
  # make bmr reproducible
  set.seed(2585)

  # ::FIXME:: coordinates ?

  # removing ens for now
  # task = dropFeatures(task, "ens")
  # removing tsa_hp1 for now
  # task = dropFeatures(task, "tsa_hp1")

  # input missing values

  # grid search for param tuning
  ctrl = makeTuneControlGrid()

  # absolute number feature selection paramset for fusing learner with the filter method
  ps = makeParamSet(makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(task))))

  # inner resampling loop
  inner = makeResampleDesc("LOO")

  # outer resampling loop
  outer = makeResampleDesc("LOO")

  # benchmarking
  res = benchmark(
    measures = list(mae, mse, rmse, timetrain),
    tasks = task,
    learners = learners,
    resamplings = outer,
    show.info = FALSE,
    models = FALSE
  )

  bmr = list(
    res = res,
    task = task
  )

}

# defining the list of hours on which we will perform the bmrs
hours = as.list(stations.dyn$mtime)[1:480]

lrns = list(
  # defining the simple learners
  lrn.lm.alt_x_y = makeFilterWrapper(
    learner = makeLearner(
      cl = "regr.lm",
      id = "multiReg.alt_x_y",
      predict.type = 'se'),
    fw.method = "linear.correlation",
    fw.mandatory.feat = c("elevation", "y", "x"),
    fw.abs = 3),

  # this learner does not support missing values. So we input these
  # lrn.lm.alt_x_y = makeImputeWrapper(
  #   lrn.lm.alt_x_y,
  #   cols = list(tsa_hp1 = imputeMedian()))

  lrn.gstat.idw = makeLearner(
    cl = "regr.gstat",
    id = "idw",
    predict.type = "se"),

  lrn.gstat.ts1 = makeLearner(
    cl = "regr.gstat",
    id = "ts1",
    par.vals = list(degree = 1),
    predict.type = "se"),

  lrn.gstat.ts2 = makeLearner(
    cl = "regr.gstat",
    id = "ts2",
    par.vals = list(degree = 2),
    predict.type = "se"),

  lrn.gstat.ok = makeFilterWrapper(
    learner = makeLearner(
      cl = "regr.gstat",
      id = "ok",
      par.vals = list(
        range = 800,
        psill = 200000,
        model.manual = "Sph",
        nugget = 0),
      predict.type = "se"),
    fw.method = "linear.correlation",
    fw.mandatory.feat = c("y", "x"),
    fw.abs = 2),

  lrn.gstat.ked = makeFilterWrapper(
    learner = makeLearner(
      cl = "regr.gstat",
      id = "ked",
      par.vals = list(
        range = 800,
        psill = 200000,
        model.manual = "Sph",
        nugget = 0),
      predict.type = "se"),
    fw.method = "linear.correlation",
    fw.mandatory.feat = c("y", "x", "elevation"),
    fw.abs = 3),

  lrn.gstat.1nn = makeFilterWrapper(
    learner = makeLearner(
      cl = "regr.gstat",
      id = "nn1",
      par.vals = list(
        nmax = 1),
      predict.type = "se"),
  fw.method = "linear.correlation",
  fw.mandatory.feat = c("y", "x"),
  fw.abs = 2)

)

# performing the benchmarks
bmrs = lapply(seq_along(hours), function(h)
  make.1h.bmr(
    task = mlr::makeRegrTask(
      id = as.character(stations.dyn[h,]$mtime),
      data = make.1h.data(stations.dyn[h,]$mtime),
      target = "tsa"
    ),
    dateTime = hours[[h]],
    learners = lrns)
)

# redefining the learners from bmr for easier access
lrns = mlr::getBMRLearners(bmr = bmr.test$res)

# extracting all the tasks from the bmrs
tasks = sapply(bmrs,function(x) x["task"])

# extracting all the bmr results from bmrs
res = sapply(bmrs,function(x) x["res"])

# merging all the benchmarks results in a single df
res = mergeBenchmarkResults(bmrs = res)

# summarizing res outputs
insights = data.frame(res) %>%
  dplyr::group_by(learner.id) %>%
  dplyr::summarise_at(
    .vars = c("mae", "mse", "rmse", "timetrain"),
    .funs = c("mean", "min", "max", "sd")) %>%
  dplyr::arrange(rmse_mean)
insights


# training a learner
m = mlr::train(
  learner = lrns$nn1.filtered,
  task = bmr.test$task)

# prediction
p = predict(m, newdata = newdata)
s = grid.sf %>%
  dplyr::bind_cols(p$data)

# quickplot
plot(s["response"])






# https://stackoverflow.com/questions/49730657/mlr-getbmrmodels-how-to-access-each-model-from-the-benchmark-result



# training
m = mlr::train(
  learner = mlr::getBMRLearners(bmr = res)[[as.character(best.learner$learner.id)]],
  task = regr.task)



# preparing the required spatial objects for mapping
data("wallonia")
ourPredictedGrid = sf::st_as_sf(s, coords = c("X", "Y"))
ourPredictedGrid = sf::st_set_crs(ourPredictedGrid, 4326)
sfgrid = sf::st_sf(sf::st_make_grid(x = sf::st_transform(wallonia, 3812),  cellsize = 1000, what = "polygons"))
ourPredictedGrid = sf::st_transform(ourPredictedGrid, crs = 3812)
ourPredictedGrid = sf::st_join(sfgrid, ourPredictedGrid)
# limit it to Wallonia
ourPredictedGrid = sf::st_intersection(ourPredictedGrid, sf::st_transform(wallonia, crs = 3812))
# stations
records.sf = sf::st_as_sf(records.1h.data, coords = c("X", "Y"))
records.sf = sf::st_set_crs(records.sf, 4326)


# Definition of the function to build a leaflet map for prediction with associated uncertainty
leafletize <- function(data.sf, borders, stations){

  # be sure we are in the proper 4326 EPSG
  data.sf = sf::st_transform(data.sf, 4326)
  stations = sf::st_transform(stations, 4326)

  # to make the map responsive
  responsiveness.chr = "\'<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\'"

  # Sometimes the interpolation and the stations don't have values in the same domain.
  # this lead to mapping inconsistency (transparent color for stations)
  # Thus we create a fullDomain which is a rowbinding of interpolated and original data
  fullDomain = c(data.sf$response, stations$tsa)

  # defining the color palette for the response
  varPal <- leaflet::colorNumeric(
    palette = "RdYlBu", #"RdBl",
    reverse = TRUE,
    domain = fullDomain, #data.sf$response,
    na.color = "transparent"
  )

  # Definition of the function to create whitening
  alphaPal <- function(color) {
    alpha <- seq(0,1,0.1)
    r <- col2rgb(color, alpha = T)
    r <- t(apply(r, 1, rep, length(alpha)))
    # Apply alpha
    r[4,] <- alpha*255
    r <- r/255.0
    codes <- (rgb(r[1,], r[2,], r[3,], r[4,]))
    return(codes)
  }

  # actually building the map
  prediction.map = leaflet::leaflet(data.sf) %>%
    # basemaps
    addProviderTiles(group = "Stamen",
      providers$Stamen.Toner,
      options = providerTileOptions(opacity = 0.25)
    ) %>%
    addProviderTiles(group = "Satellite",
      providers$Esri.WorldImagery,
      options = providerTileOptions(opacity = 1)
    ) %>%
    # centering the map
    fitBounds(sf::st_bbox(data.sf)[[1]],
      sf::st_bbox(data.sf)[[2]],
      sf::st_bbox(data.sf)[[3]],
      sf::st_bbox(data.sf)[[4]]
    ) %>%
    # adding layer control button
    addLayersControl(baseGroups = c("Stamen", "Satellite"),
      overlayGroups = c("prediction", "se", "Stations", "Admin"),
      options = layersControlOptions(collapsed = TRUE)
    ) %>%
    # fullscreen button
    addFullscreenControl() %>%
    # location button
    addEasyButton(easyButton(
      icon = "fa-crosshairs", title = "Locate Me",
      onClick = JS("function(btn, map){ map.locate({setView: true}); }"))) %>%
    htmlwidgets::onRender(paste0("
      function(el, x) {
      $('head').append(",responsiveness.chr,");
      }")
    ) %>%
    # predictions
    addPolygons(
      group = "prediction",
      color = "#444444", stroke = FALSE, weight = 1, smoothFactor = 0.8,
      opacity = 1.0, fillOpacity = 0.9,
      fillColor = ~varPal(response),
      highlightOptions = highlightOptions(color = "white", weight = 2,
        bringToFront = TRUE),
      label = ~htmltools::htmlEscape(as.character(response))
    ) %>%
    addLegend(
      position = "bottomright", pal = varPal, values = ~response,
      title = "prediction",
      group = "prediction",
      opacity = 1
    )

  # if se.bool = TRUE
  if (!is.null(data.sf$se)) {
    uncPal <- leaflet::colorNumeric(
      palette = alphaPal("#5af602"),
      domain = data.sf$se,
      alpha = TRUE
    )

    prediction.map = prediction.map %>%
      addPolygons(
        group = "se",
        color = "#444444", stroke = FALSE, weight = 1, smoothFactor = 0.5,
        opacity = 1.0, fillOpacity = 1,
        fillColor = ~uncPal(se),
        highlightOptions = highlightOptions(color = "white", weight = 2,
          bringToFront = TRUE),
        label = ~ paste("prediction:", signif(data.sf$response, 2), "\n","se: ", signif(data.sf$se, 2))
      ) %>%
      addLegend(
        group = "se",
        position = "bottomleft", pal = uncPal, values = ~se,
        title = "se",
        opacity = 1
      )
  }

  prediction.map = prediction.map %>%
    # admin boundaries
    addPolygons(
      data = borders,
      group = "Admin",
      color = "#444444", weight = 1, smoothFactor = 0.5,
      opacity = 1, fillOpacity = 0, fillColor = FALSE) %>%
    # stations location
    addCircleMarkers(
      data = stations,
      group = "Stations",
      color = "black",
      weight = 2,
      fillColor = ~varPal(tsa),
      stroke = TRUE,
      fillOpacity = 1,
      label = ~htmltools::htmlEscape(as.character(tsa)))

  return(prediction.map)
}

# creating the interactive map
predicted_map = leafletize(
  data.sf = ourPredictedGrid,
  borders = wallonia,
  stations = records.sf
)
predicted_map



