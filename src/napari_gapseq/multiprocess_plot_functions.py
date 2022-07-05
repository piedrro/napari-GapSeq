







def get_gapseq_compute_iterator(self, layer):
    meta = self.box_layer.metadata.copy()

    bounding_box_centres = meta["bounding_box_centres"]
    bounding_box_class = meta["bounding_box_class"]
    bounding_box_size = meta["bounding_box_size"]

    bounding_boxes = self.box_layer.data.copy()

    image = self.viewer.layers[layer].data

    compute_iterator = []

    for frame in range(image.shape[0]):

        for i in range(len(bounding_boxes)):
            box = bounding_boxes[i].tolist()
            box_centre = bounding_box_centres[i]
            box_class = bounding_box_class[i]

            mp_dict = dict(frame=frame, box=box, box_centre=box_centre, box_class=box_class,
                           box_size=bounding_box_size, image_shape=image.shape, image_type=image.dtype,
                           box_index=i, layer=layer)

            compute_iterator.append(mp_dict)

    compute_iterator = np.array_split(compute_iterator, image.shape[0])

    return compute_iterator


def process_localisation_data(self, localisation_data):
    data = []

    for i in range(len(localisation_data)):

        for j in range(len(localisation_data[0])):
            data.append(localisation_data[i][j])

    localisation_data = pd.DataFrame(data)

    localisation_data = localisation_data.groupby(["box_index"])

    localisation_dict = {}

    layer = localisation_data.get_group(list(localisation_data.groups)[0])["layer"].unique()[0]

    localisation_dict[layer] = {}

    for i in range(len(localisation_data)):
        data = localisation_data.get_group(list(localisation_data.groups)[i])

        data = data.sort_values(["frame", "box_index"])

        localisation_dict[layer][i] = {"box_mean": data.box_mean.tolist(),
                                       "box_mean_local_background": data.box_mean_local_background.tolist(),
                                       "box_mean_global_background": data.box_mean_global_background.tolist(),
                                       "box_std": data.box_std.tolist(),
                                       "box_std_local_background": data.box_std_local_background.tolist(),
                                       "box_std_global_background": data.box_std_global_background.tolist(),
                                       "gaussian_height": data.gaussian_height.tolist(),
                                       "gaussian_x": data.gaussian_x.tolist(),
                                       "gaussian_y": data.gaussian_y.tolist(),
                                       "gausian_width": data.gausian_width.tolist(),
                                       "image_shape": data.image_shape.tolist()[0],
                                       "bounding_box": data.box.tolist()[0]}

    return localisation_dict



def gapseq_compute_traces(self, progress_callback, layer):

    bounding_boxes = self.box_layer.data.copy()
    meta = self.box_layer.metadata.copy()

    bounding_box_centres = meta["bounding_box_centres"]
    bounding_box_class = meta["bounding_box_class"]
    bounding_box_size = meta["bounding_box_size"]

    image = self.viewer.layers[layer].data
    background_image, masked_image = self.get_background_mask(bounding_boxes, bounding_box_size, bounding_box_centres, image)

    shared_image_object = shared_memory.SharedMemory(create=True, size=image.nbytes)
    shared_image = np.ndarray(image.shape, dtype=image.dtype, buffer=shared_image_object.buf)
    shared_image[:] = image[:]

    shared_background_object = shared_memory.SharedMemory(create=True, size=background_image.nbytes)
    shared_background = np.ndarray(background_image.shape, dtype=background_image.dtype, buffer=shared_background_object.buf)
    shared_background[:] = background_image[:]

    box_min = self.localisation_area_min.value()
    box_max = self.localisation_area_max.value()
    threshold = self.localisation_threshold.value()

    compute_iterator = self.get_gapseq_compute_iterator(layer)

    with Pool() as p:

        def callback(*args):
            iter.append(1)
            progress = (len(iter) / len(compute_iterator)) * 100

            if progress_callback != None:
                progress_callback.emit(progress)

            return

        iter = []

        results = [p.apply_async(compute_box_stats, args=(i,), kwds={'shared_image_object': shared_image_object,
                                                                      'shared_background_object': shared_background_object,
                                                                      'threshold':threshold,
                                                                      'box_min':box_min,
                                                                      'box_max':box_max}, callback=callback) for i in compute_iterator]

        localisation_data = [r.get() for r in results]

        localisation_data = self.process_localisation_data(localisation_data)

        return localisation_data


def process_gapseq_compute_traces(self, localisation_data):

    meta = self.box_layer.metadata.copy()

    if "localisation_data" in meta.keys():

        for layer,data in localisation_data.items():

            meta["localisation_data"][layer] = data

    else:
        meta["localisation_data"] = localisation_data

    layer_image_shape = {}
    bounding_box_breakpoints = {}
    bounding_box_traces = {}

    image_layers = [layer.name for layer in self.viewer.layers if
                    layer.name not in ["bounding_boxes", "localisation_threshold"]]

    for layer in image_layers:

        image = self.viewer.layers[layer].data
        data = localisation_data[layer][0]["box_mean"]
        layer_image_shape[layer] = image.shape

        bounding_box_breakpoints[layer] = []
        bounding_box_traces[layer] = []

        for i in range(len(localisation_data[layer])):

            bounding_box_breakpoints[layer].append([])
            bounding_box_traces[layer].append([0] * len(data))

    meta["layer_image_shape"] = layer_image_shape
    meta["bounding_box_breakpoints"] = bounding_box_breakpoints
    meta["bounding_box_traces"] = bounding_box_traces

    self.box_layer.metadata = meta
    self.plot_compute_progress.setValue(0)

    self.plot_graphs()
    self.plot_fit_graph()


def initalise_gapseq_compute_traces(self):

    image_layers = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "localisation_threshold"]]

    for layer in image_layers:

        worker = Worker(partial(self.gapseq_compute_traces, layer = layer))
        worker.signals.result.connect(self.process_gapseq_compute_traces)
        worker.signals.progress.connect(partial(self.gapseq_progressbar, progressbar="compute"))
        self.threadpool.start(worker)




def threshold_localisation_image(image, threshold, box_min, box_max):

    img = image

    img = difference_of_gaussians(img, 1)

    img = normalize99(img)
    img = rescale01(img) * 255

    img = img.astype(np.uint8)

    _, mask = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

    footprint = disk(1)
    mask = erosion(mask, footprint)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=cv.contourArea, reverse=True)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > box_min and cv.contourArea(cnt) < box_max]

    mask = np.zeros_like(img, dtype=np.uint8)
    mask = cv.drawContours(mask, contours, -1, 255, -1)

    return mask


def compute_box_stats(box_data, shared_image_object, shared_background_object, threshold, box_min, box_max):

    try:

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            frame = box_data[0]["frame"]

            image_shm = shared_memory.SharedMemory(
                name=shared_image_object.name)
            image = np.ndarray(
                box_data[0]["image_shape"], dtype=box_data[0]["image_type"], buffer=image_shm.buf)[frame]

            background_shm = shared_memory.SharedMemory(
                name=shared_background_object.name)
            background_image = np.ndarray(
                box_data[0]["image_shape"], dtype=box_data[0]["image_type"], buffer=background_shm.buf)[frame]

            localisation_mask = threshold_localisation_image(image, threshold, box_min, box_max)

            for i in range(len(box_data)):

                box = box_data[i]["box"]
                [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = box

                cx, cy = box_data[i]["box_centre"]
                box_class = box_data[i]["box_class"]
                box_size = box_data[i]["box_size"]

                background_box_size = 10
                img = image[int(y1):int(y2), int(x1):int(x2)].copy()
                mask = localisation_mask[int(y1):int(y2), int(x1):int(x2)].copy()

                [[by1,bx1],[by2,bx2]] = [[cy - background_box_size, cx - background_box_size],
                                         [cy + background_box_size, cx + background_box_size]]

                local_background_data = background_image[int(by1):int(by2), int(bx1):int(bx2)].copy()

                box_data[i]["box_mean"] = np.nanmean(img)
                box_data[i]["box_mean_global_background"] = np.nanmean(background_image)
                box_data[i]["box_mean_local_background"] = np.nanmean(local_background_data)

                box_data[i]["box_std"] = np.nanstd(img)
                box_data[i]["box_std_global_background"] = np.nanstd(background_image)
                box_data[i]["box_std_local_background"] = np.nanstd(local_background_data)

                # params, success = fitgaussian(img)
                params = [0,0,0,0,0]
                success = 5

                gaussian_x = cx + 1 - box_size + params[2]
                gaussian_y = cy + 1 - box_size + params[1]

                fit_error = False
                if gaussian_x < x1 or gaussian_x > x2:
                    fit_error = True
                if gaussian_y < y1 or gaussian_y > y2:
                    fit_error = True
                if success != 1:
                    fit_error = True
                if np.max(mask) == 0:
                    fit_error = True

                if fit_error is True:
                    params = [0,0,0,0,0]
                    gaussian_x = cy + 1 - box_size + params[1]
                    gaussian_y = cx + 1 - box_size + params[2]

                box_data[i]["box_class"] = box_class
                box_data[i]["gaussian_height"] = params[0]
                box_data[i]["gaussian_x"] = gaussian_x
                box_data[i]["gaussian_y"] = gaussian_y
                box_data[i]["gausian_width"] = params[3]

            del image
            del background_image
            image_shm.close()
            background_shm.close()


    except:
        import traceback
        print(traceback.format_exc())
        pass

    return box_data
