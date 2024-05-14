import os
def validate_args(args):
    full_path = os.path.join('data', args.data_dir)

    # Check if the directory exists
    if not os.path.isdir(full_path):
        raise FileNotFoundError(f"The directory specified does not exist: {full_path}")

    # Check if the specified database is correct
    if args.database not in ['RadImageNet', 'ImageNet']:
        raise Exception('Pre-trained database does not exist. Please choose ImageNet or RadImageNet.')

    # Check if the structure argument is valid
    if args.structure not in ['unfreezeall', 'freezeall', 'unfreezetop10']:
        raise Exception('Invalid structure option. Choose to unfreezeall, freezeall, or unfreezetop10 layers.')