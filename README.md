# OptimallyOrientedFlux
# Implementation of an OOF filter

    import SimpleITK as sitk
    import OOF

    class Options:
        radii = list(np.round(np.linspace(0.7, 6.0, 10, endpoint=True), 2))
        type = 3

# open image using SITK
    folder = '...'
    name = 'image.nii.gz'
    input_image = sitk.ReadImage(os.path.join(folder, name))

# OOF filter
    oof_image = OOF.oof3response(image=input_image, radii=options.radii, resp_type=options.type)
