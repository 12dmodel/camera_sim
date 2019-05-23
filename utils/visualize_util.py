import os
import shutil
import skimage
import numpy as np
from . import HTML


def transpose_table(table):
    return [[table[j][i] for j in range(len(table))] for i in range(len(table[0]))]


def write_table_of_images(path, table, img_folder='', metastr='', compare_col=None, path_to_resource=None):
    """
    Args:
        path: path to write html file.
        table: list of lists that represent table to be written.
        img_folder: folder to write image files to.
        metastr: extra string to append after the table.
        compare_col: columns to add sliding comparison.
        path_to_resource: path to javascript resource for doing comparison
            (e.g. /media/nfs/www/resources/twentytwenty)
    """
    # go through the table and convert everything to string.
    img_template = '<img width="100%" src="{}" />'
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    n_img = 0
    for row_idx, row in enumerate(table):
        compare_idx = []
        diff_img = None
        for i, cell in enumerate(row):
            if not isinstance(cell, str):
                # convert it to image
                img_name = 'img{}.png'.format(n_img)
                skimage.io.imsave(os.path.join(dirname, img_name),
                                  cell)
                row[i] = img_template.format(os.path.join(img_folder, img_name))
                if compare_col is not None and i in compare_col:
                    compare_idx.append(n_img)
                    if diff_img is None:
                        diff_img = cell.astype('float')
                    else:
                        diff_img -= cell.astype('float')
                        diff_img = (diff_img + 127.5) * 0.5
                        img_name = 'diff{}.png'.format(row_idx)
                        diff_img = np.clip(diff_img, 0, 255)
                        skimage.io.imsave(os.path.join(dirname, img_name),
                                          diff_img.astype('uint8'))
                        # this will cause a crash if more images is to be added.
                        diff_img = 'haha'
                        
                n_img += 1
        if compare_col is not None:
            if row_idx == 0:
                row.append("Comparison of column {}".format(compare_col))
                row.append("Diff of column {}".format(compare_col))
            else:
                div =  '<div class="twentytwenty-container">\n'
                for idx in compare_idx:
                    img_name = 'img{}.png'.format(idx)
                    div += img_template.format(os.path.join(img_folder, img_name)) + '\n'
                div += '</div>\n'
                row.append(div)
                # Add diff image.
                img_name = 'diff{}.png'.format(row_idx)
                row.append(img_template.format(os.path.join(img_folder, img_name)))
    if compare_col is not None and path_to_resource is not None:
        try:
            dst = os.path.dirname(path)
            shutil.copytree(os.path.join(path_to_resource, 'js'), os.path.join(dst, 'js'))
            shutil.copytree(os.path.join(path_to_resource, 'css'), os.path.join(dst, 'css'))
        except FileExistsError as e:
            pass

    table_html = str(HTML.Table(table, width="100%"))

    with open(path, 'w') as f:
        f.write("<html>\n"
                "  <head>\n"
                '    <link href="css/foundation.css" rel="stylesheet" type="text/css" />\n'
                '    <link href="css/twentytwenty.css" rel="stylesheet" type="text/css" />\n'
                '    <style>\n'
                '      table {\n'
                '        table-layout: fixed;\n'
                '      }\n'
                '    </style>\n'
                "  </head>\n"
                "  <body>\n")
        f.write(table_html)
        f.write('\n')
        f.write(metastr)
        if compare_col is not None:
            f.write('<script \n'
                    'src="https://code.jquery.com/jquery-3.2.1.js"\n'
                    'integrity="sha256-DZAnKJ/6XZ9si04Hgrsxu/8s717jcIzLy3oi35EouyE="\n'
                    'crossorigin="anonymous"></script>\n'
                    '<script src="js/jquery.event.move.js"></script>\n'
                    '<script src="js/jquery.twentytwenty.js"></script>\n'
                    '<script>\n'
                    '$(function(){\n'
                    '  $(".twentytwenty-container[data-orientation!=\'vertical\']").twentytwenty({default_offset_pct: 0.7});\n'
                    '  $(".twentytwenty-container[data-orientation=\'vertical\']").twentytwenty({default_offset_pct: 0.3, orientation: \'vertical\'});\n'
                    '});\n'
                    '</script>\n\n')
        f.write("  </body>\n"
                "</html>")

