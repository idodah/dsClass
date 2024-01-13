import pkg_resources
def get_file_path(file_name):
    data_path = pkg_resources.resource_filename('dsClass', r'data/' + file_name)
    return data_path
