"""Run this file, then choose 0 for the complete automatic viva demo."""
import sys
import numpy as np
import matrix_ops as ops
import transformations as tr
import analysis as la
import visualize as vis

np.set_printoptions(precision=4, suppress=True)
SHAPES = {
    'triangle': np.array([[0.,2.,1.],[0.,0.,2.]]),
    'square': np.array([[0.,2.,2.,0.],[0.,0.,2.,2.]]),
    'rectangle': np.array([[0.,4.,4.,0.],[0.,0.,2.,2.]]),
    'house': np.array([[0.,2.,2.,1.,0.],[0.,0.,2.,3.,2.]])}


def numbers(prompt, count=None):
    return numbers_from(input(prompt), count)


def numbers_from(text, count=None):
    values = np.array([float(x) for x in text.replace(',',' ').split()])
    if len(values) == 0 or not np.isfinite(values).all() or (count is not None and len(values) != count):
        raise ValueError(f'Enter {count if count is not None else "one or more"} finite numbers.')
    return values


def read_matrix():
    print('Enter rows separated by semicolons, e.g. 1 2; 3 4')
    return ops.matrix([[float(x) for x in row.replace(',',' ').split()]
                       for row in input('Matrix: ').split(';')])


def choose_shape():
    name = input('Shape (triangle/square/rectangle/house/custom): ').strip().lower()
    if name in SHAPES:
        return SHAPES[name].copy()
    if name != 'custom':
        raise ValueError('Choose one of the listed shapes.')
    print('Enter boundary vertices in order, one point per line; avoid crossing edges.')
    print('Type "u" to remove the last point, or press Enter on an empty line to finish.')
    points = []
    live = vis.LiveShape()
    try:
        while True:
            text = input(f'Point {len(points)+1} (x y): ').strip().lower()
            if text == '':
                if len(points) >= 3: break
                print('A shape needs at least three points.'); continue
            if text == 'u':
                if points: print('Removed', points.pop())
            else:
                try: points.append(tuple(float(v) for v in numbers_from(text, 2)))
                except ValueError as error: print('Please try again:', error); continue
            live.update(points)
    finally:
        live.close()
    return np.array(points).T


def choose_transform(allow_translation=True):
    name = input('Operation (translate/scale/scale_xy/rotate/reflect/shear/project): ').strip().lower()
    if name == 'translate':
        if not allow_translation:
            raise ValueError('Choose a linear transformation for this feature.')
        return name, tr.translate(*numbers('tx ty: ',2))
    if name == 'scale': return name,tr.scale(numbers('k: ',1)[0])
    if name == 'scale_xy': return name,tr.scale_xy(*numbers('sx sy: ',2))
    if name == 'rotate': return name,tr.rotate(numbers('Angle in degrees: ',1)[0])
    if name == 'reflect': return name,tr.reflect(input('Axis (x/y/y=x): ').strip())
    if name == 'shear': return name,tr.shear(*numbers('shx shy: ',2))
    if name == 'project': return name,tr.project_onto_x()
    raise ValueError('Choose a listed operation.')


def transform(shape, name, T, filename=None):
    if name == 'translate':
        offset = np.repeat(T,shape.shape[1],axis=1)
        print('Shape:\n',shape,'\nTranslation offset:\n',offset)
        result = ops.add(shape,offset)
        print('Translation has no 2x2 T; its linear part is I (det=1, rank=2).')
        ops.verified(la.area(shape),la.area(result))
        vis.plot_shape(shape,result,'Translation: addition; linear part I',np.eye(2),filename)
    else:
        result = tr.apply(T,shape)
        la.det_meaning(T,shape); ops.rank(T)
        vis.plot_shape(shape,result,name,T,filename)
    return result


def calculator():
    name = input('Calculator (add/sub/scalar/mul/transpose/det/rank/inverse): ').strip()
    A = read_matrix()
    print('A:\n',A)
    if name in ('add','sub','mul'):
        B = read_matrix(); print('B:\n',B)
        {'add':ops.add,'sub':ops.subtract,'mul':ops.multiply}[name](A,B)
    elif name == 'scalar': ops.scalar_multiply(A,numbers('Scalar: ',1)[0])
    elif name in ('transpose','det','rank','inverse'):
        {'transpose':ops.transpose,'det':ops.determinant,'rank':ops.rank,
         'inverse':ops.inverse_gauss_jordan}[name](A)
    else: raise ValueError('Choose a listed calculator operation.')


def inverse(shape, A=None, filename=None):
    if A is None:
        A = read_matrix()
    print('A:\n', A)
    steps, result = ops.gauss_jordan_steps(A)
    vis.gauss_jordan(steps, result, shape, filename)
    return result


def demo():
    print('\nFULL DEMO: 2D Shape Transformer – Matrix Operations in Computer Graphics')
    shape = SHAPES['house'].copy()
    for name, points in SHAPES.items():
        print('Built-in shape:',name,'\n',points)
    print('Custom example:\n',np.array([[0,3,2,0],[0,0,2,1]]))
    operations = [('translate',tr.translate(2,1)),('scale',tr.scale(2)),
                  ('scale_xy',tr.scale_xy(2,.5)),('rotate',tr.rotate(30)),
                  ('reflect_x',tr.reflect('x')),('reflect_y',tr.reflect('y')),
                  ('reflect_diagonal',tr.reflect('y=x')),('shear',tr.shear(1,.25)),
                  ('project',tr.project_onto_x())]
    for name,T in operations:
        result = transform(shape,name,T,name+'.png')
        if name == 'translate':
            print('Undo translation by subtracting offset.')
            restored = ops.subtract(result,np.repeat(T,shape.shape[1],axis=1))
        else:
            restored = tr.undo(T,result)
        if restored is not None: ops.verified(restored,shape)
    R = tr.rotate(30); tr.rotation_orthogonal(R)
    first, second = vis.chain(shape,R,tr.scale_xy(2,1),'chain.png')
    ops.verified(np.allclose(first,second),False)
    la.find_transformation([1,0],[0,1],[2,1],[1,3])
    la.find_transformation([1,0],[2,0],[1,0],[2,0])
    la.find_transformation([1,0],[2,0],[1,0],[3,0])
    la.subspaces(tr.project_onto_x()); la.subspaces(tr.shear(1,0))
    la.eigen(tr.scale_xy(2,-1),'eigenvectors.png'); la.eigen(R)
    la.svd(tr.shear(1,.25),shape,'svd.png')
    A = np.array([[2.,1.],[1.,3.]])
    B = np.array([[1.,2.],[3.,4.]])
    ops.add(A,B); ops.subtract(A,B); ops.scalar_multiply(A,3)
    ops.multiply(A,B); ops.transpose(A); ops.determinant(A); ops.rank(A)
    ops.determinant([[1,2,3],[0,4,5],[1,0,6]])
    ops.inverse_gauss_jordan(A); ops.inverse_gauss_jordan([[1,2],[2,4]])
    inverse(shape,A,'inverse.png')
    print('Demo complete. Plots saved in outputs/.')


def main():
    shape = SHAPES['square'].copy()
    last = None
    while True:
        print('\n1. Choose shape\n2. Apply a transformation\n3. Chain two transformations (AB vs BA)'
              '\n4. Undo last transformation\n5. Find transformation from point pairs'
              '\n6. Four fundamental subspaces\n7. Eigenvectors \n8. SVD'
              '\n9. Basic matrix calculator\n10. Inverse (Gauss–Jordan, step by step)\n0. Run full demo\nq. Quit')
        try:
            choice = input('Choice: ').strip().lower()
            if choice == 'q': break
            if choice == '0': demo()
            elif choice == '1':
                shape = choose_shape(); last = None; print('Shape:\n',shape)
                vis.show_shape(shape,'Chosen shape')
            elif choice == '2':
                name,T = choose_transform()
                result = transform(shape,name,T)
                last = (name,T,shape.copy()); shape = result
            elif choice == '3':
                print('Choose A:'); _,A = choose_transform(False)
                print('Choose B:'); _,B = choose_transform(False)
                vis.chain(shape,A,B)
            elif choice == '4':
                if last is None: print('Apply a transformation first.'); continue
                name,T,original = last
                if name == 'translate': result = ops.subtract(shape,np.repeat(T,shape.shape[1],axis=1))
                else: result = tr.undo(T,shape)
                if result is not None:
                    ops.verified(result,original)
                    vis.plot_shape(shape,result,'Undo last transformation')
                    shape = result; last = None
            elif choice == '5':
                la.find_transformation(numbers('p1 (x y): ',2),numbers('p2 (x y): ',2),
                                       numbers('q1 (x y): ',2),numbers('q2 (x y): ',2))
            elif choice in ('6','7','8'):
                _,T = choose_transform(False)
                if choice == '6': la.subspaces(T)
                elif choice == '7': la.eigen(T)
                else: la.svd(T,shape)
            elif choice == '9': calculator()
            elif choice == '10': inverse(shape)
            else: print('Please choose 0–10 or q.')
        except (EOFError,KeyboardInterrupt):
            print('\nGoodbye.'); break
        except (ValueError,TypeError,OverflowError,np.linalg.LinAlgError) as error:
            print('Please try again:',error)
        except OSError as error:
            print('Could not open/save the plot:',error)


if __name__ == '__main__':
    if '--demo' in sys.argv: demo()
    else: main()
