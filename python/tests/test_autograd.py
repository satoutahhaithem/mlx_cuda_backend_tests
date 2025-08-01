# Copyright © 2023 Apple Inc.

import gc
import unittest

import mlx.core as mx
import mlx_tests


class TestAutograd(mlx_tests.MLXTestCase):
    def test_jvp(self):
        fun = lambda x: 2 * x
        out, dout = mx.jvp(fun, [mx.array(1.0)], [mx.array(2.0)])
        self.assertEqual(out[0].item(), 2.0)
        self.assertEqual(dout[0].item(), 4.0)

        fun = lambda x, y: x * y
        _, out = mx.jvp(
            fun, [mx.array(4.0), mx.array(2.0)], [mx.array(3.0), mx.array(2.0)]
        )
        self.assertEqual(out[0].item(), 4.0 * 2.0 + 2.0 * 3.0)

        fun = lambda x, y, z: (x * y, y * z)
        _, out = mx.jvp(
            fun,
            [mx.array(2.0), mx.array(4.0), mx.array(6.0)],
            [mx.array(1.0), mx.array(3.0), mx.array(1.0)],
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].item(), 4.0 * 1.0 + 2.0 * 3.0)
        self.assertEqual(out[1].item(), 4.0 * 1.0 + 6.0 * 3.0)

    def test_vjp(self):
        fun = lambda x: 2 * x
        out, dout = mx.vjp(fun, [mx.array(1.0)], [mx.array(2.0)])
        self.assertEqual(out[0].item(), 2.0)
        self.assertEqual(dout[0].item(), 4.0)

        fun = lambda x, y: x * y
        _, dout = mx.vjp(fun, [mx.array(4.0), mx.array(2.0)], [mx.array(3.0)])
        self.assertEqual(dout[0].item(), 6.0)
        self.assertEqual(dout[1].item(), 12.0)

        fun = lambda x, y, z: (x * y, y * z)
        _, out = mx.vjp(
            fun,
            [mx.array(2.0), mx.array(4.0), mx.array(6.0)],
            [mx.array(1.0), mx.array(3.0)],
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].item(), 4.0 * 1.0)
        self.assertEqual(out[1].item(), 2.0 * 1.0 + 6.0 * 3.0)
        self.assertEqual(out[2].item(), 4.0 * 3.0)

    def test_grad(self):
        fun = lambda x: x * x

        value, dfdx = mx.value_and_grad(fun)(mx.array(0.5))
        self.assertEqual(value.item(), 0.25)
        self.assertEqual(dfdx.item(), 1.0)

        dfdx = mx.grad(fun)(mx.array(0.5))
        self.assertEqual(dfdx.item(), 1.0)

        df2dx2 = mx.grad(mx.grad(fun))(mx.array(0.5))
        self.assertEqual(df2dx2.item(), 2.0)
        df3dx3 = mx.grad(mx.grad(mx.grad(fun)))(mx.array(0.5))
        self.assertEqual(df3dx3.item(), 0.0)

        fun = lambda x, y: x * y
        x = mx.array(2.0)
        y = mx.array(3.0)
        dfdx = mx.grad(fun, argnums=0)(x, y)
        self.assertEqual(dfdx.item(), 3.0)
        dfdx = mx.grad(fun, argnums=1)(x, y)
        self.assertEqual(dfdx.item(), 2.0)

        # Pass non array args to functions works
        fun = lambda x, y: x
        value, dfdx = mx.value_and_grad(fun)(mx.array(2.0), "hello")
        self.assertEqual(value.item(), 2.0)
        self.assertEqual(dfdx.item(), 1.0)

        dfdx = mx.grad(fun)(mx.array(2.0), "hello")
        self.assertEqual(dfdx.item(), 1.0)

        # Raises when function does not return array
        fun = lambda x: "hello"
        with self.assertRaises(ValueError):
            mx.grad(fun)(mx.array(2.0))

        # Raises for invalid argument number or argument type
        fun = lambda x: x
        with self.assertRaises(ValueError):
            mx.grad(fun, argnums=2)(mx.array(2.0))
        with self.assertRaises(ValueError):
            mx.grad(fun, argnums=-2)(mx.array(2.0))
        with self.assertRaises(ValueError):
            mx.grad(fun)("hello")

        # Raises when output is not a scalar array
        fun = lambda x: mx.sum(x, keepdims=True)
        with self.assertRaises(ValueError):
            mx.grad(fun)(mx.ones((2, 2)))

    def test_grad_trees(self):
        fun = lambda x, y: x * y
        value, dfdx = mx.value_and_grad(fun, (0, 1))(mx.array(0.5), mx.array(2.0))
        self.assertEqual(value.item(), 1.0)
        self.assertTrue(isinstance(dfdx, tuple))
        self.assertEqual(dfdx[0].item(), 2.0)
        self.assertEqual(dfdx[1].item(), 0.5)

        fun = lambda x, y: x * y
        value, dfdx = mx.value_and_grad(fun, 1)(mx.array(0.5), mx.array(2.0))
        self.assertEqual(value.item(), 1.0)
        self.assertEqual(dfdx.item(), 0.5)

        fun = lambda p: p["x"] * p["y"]
        value, dfdx = mx.value_and_grad(fun)({"x": mx.array(0.5), "y": mx.array(2.0)})
        self.assertEqual(value.item(), 1.0)
        self.assertEqual(dfdx["x"].item(), 2.0)
        self.assertEqual(dfdx["y"].item(), 0.5)

        fun = lambda p: p["x"] * p["y"]
        with self.assertRaises(ValueError):
            mx.value_and_grad(fun)({"x": 0.5, "y": mx.array(2.0)})
        with self.assertRaises(ValueError):
            mx.value_and_grad(fun, (0, 1))({"x": mx.array(0.5), "y": mx.array(2.0)})

        fun = lambda p, b: mx.square(p[0]["foo"][2]) * b
        value, dfdx = mx.value_and_grad(fun)(
            [{"foo": [[], [], mx.array(2.0)]}], mx.array(0.5)
        )
        self.assertEqual(value.item(), 2.0)
        self.assertEqual(dfdx[0]["foo"][2].item(), 2.0)

        fun = lambda x: x
        with self.assertRaises(TypeError):
            mx.value_and_grad(fun, (None, None))
        with self.assertRaises(ValueError):
            mx.value_and_grad(fun, tuple())
        with self.assertRaises(ValueError):
            mx.grad(fun, argnums=(0, 0))

    def test_auxiliary_values(self):
        def fun(x, y):
            l = (x * y).sum()
            extra = {"loss": l, "foo": y.square() + x.square(), "bar": [1, 2, 3, y, x]}
            return l, extra

        fun_value_grad = mx.value_and_grad(fun)
        fun_grad = mx.grad(fun)

        (loss, a), b = fun_value_grad(mx.ones((2, 2)), mx.ones((2, 2)))
        self.assertEqual(a["loss"].item(), 4)
        self.assertTrue(mx.array_equal(b, mx.ones((2, 2))))
        self.assertTrue(mx.array_equal(a["foo"], 2 * mx.ones((2, 2))))
        self.assertEqual(a["bar"][:3], [1, 2, 3])
        self.assertTrue(mx.array_equal(a["bar"][3], mx.ones((2, 2))))
        self.assertTrue(mx.array_equal(a["bar"][4], mx.ones((2, 2))))

        with self.assertRaises(ValueError):
            _ = fun_grad(mx.ones((2, 2)), mx.ones((2, 2)))

    def test_grad_kwargs(self):
        fun = lambda x, y: x * y
        a, b = mx.array(0.5), mx.array(2.0)
        dfdx = mx.grad(fun)
        self.assertEqual(dfdx(a, b).item(), 2.0)
        self.assertEqual(dfdx(a, y=b).item(), 2.0)
        with self.assertRaises(ValueError):
            dfdx(x=a, y=b).item()

        dfdy = mx.grad(fun, argnums=[], argnames=["y"])
        with self.assertRaises(ValueError):
            dfdy(a, b)
        grads = dfdy(a, y=b)
        self.assertTrue(isinstance(grads, tuple))
        self.assertTrue(grads[0] is None)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["y"].item(), 0.5)
        grads = dfdy(x=a, y=b)
        self.assertEqual(grads[1]["y"].item(), 0.5)
        self.assertEqual(len(grads[1]), 1)

        dfdxy = mx.grad(fun, argnums=[0], argnames=["y"])
        with self.assertRaises(ValueError):
            dfdxy(a, b)
        with self.assertRaises(ValueError):
            dfdxy(x=a, y=b)
        grads = dfdxy(a, y=b)
        self.assertTrue(isinstance(grads, tuple))
        self.assertEqual(grads[0].item(), 2.0)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["y"].item(), 0.5)

        fun = lambda x, y, z: x * y * z
        dfdxyz = mx.grad(fun, argnums=[0, 1], argnames=["z"])
        c = mx.array(4.0)
        grads = dfdxyz(a, b, z=c)
        self.assertTrue(isinstance(grads, tuple))
        self.assertTrue(isinstance(grads[0], tuple))
        self.assertEqual(grads[0][0].item(), 8.0)
        self.assertEqual(grads[0][1].item(), 2.0)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["z"].item(), 1.0)

        fun = lambda x, y: x * y
        dfdy = mx.grad(fun, argnames=["y"])
        grads = dfdy(a, y=b)
        self.assertTrue(isinstance(grads, tuple))
        self.assertTrue(grads[0] is None)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["y"].item(), 0.5)

    def test_captured(self):
        a = mx.array(5.0)
        f = lambda x: a + x
        g = lambda x: a + a
        h = lambda x: x + x

        dfdx = mx.grad(f)
        self.assertEqual(dfdx(a).item(), 1.0)

        dgdx = mx.grad(g)
        self.assertEqual(dgdx(a).item(), 0.0)

        dhdx = mx.grad(h)
        self.assertEqual(dhdx(a).item(), 2.0)

        d2fdx2 = mx.grad(dfdx)
        self.assertEqual(d2fdx2(a).item(), 0.0)

        d2gdx2 = mx.grad(dgdx)
        self.assertEqual(d2gdx2(a).item(), 0.0)

        d2hdx2 = mx.grad(dhdx)
        self.assertEqual(d2hdx2(a).item(), 0.0)

    def test_stop_gradient(self):
        shape_in = (4, 4)
        w_in = mx.ones(shape_in)
        x_in = mx.ones(shape_in)
        cotan = mx.ones(shape_in)

        def h(w, x):
            x1 = 2 * x
            y = mx.stop_gradient(x1)
            y1 = 3 * y
            return w @ y1

        vals, vjps = mx.vjp(h, [w_in, x_in], [cotan])
        mx.eval(vjps)

        self.assertTrue(mx.allclose(vjps[0], 24.0 * mx.ones(shape_in)))
        self.assertTrue(mx.allclose(vjps[1], mx.zeros(shape_in)))

        g = lambda x: h(w_in, x)
        vals, vjps = mx.vjp(g, [x_in], [cotan])
        mx.eval(vjps)

        self.assertTrue(mx.allclose(vjps[0], mx.zeros(shape_in)))

    def test_update_state(self):
        y = mx.array([1.0])
        state = mx.zeros((2,))

        def fn(y, x):
            nonlocal state
            x = y * x
            state = state + x
            return x.sum()

        x = mx.ones((2,))
        mx.grad(fn)(y, x)
        mx.eval(state)
        self.assertTrue(mx.allclose(state, mx.ones((2,))))

    def test_scatter_vjp(self):
        def fun(x, idx):
            x[idx] = 2.0
            return x.sum()

        dfdx = mx.grad(fun)(mx.array([1.0, 2.0, 3.0]), mx.array([1]))
        self.assertTrue(mx.array_equal(dfdx, mx.array([1.0, 0.0, 1.0])))
        self.assertEqual(dfdx.dtype, mx.float32)

        y = mx.array([0.0, 1.0, 2.0])

        def fun(x, idx):
            y[idx] = x
            return y.sum()

        dfdx = mx.grad(fun)(mx.array([2.0]), mx.array([1]))
        self.assertTrue(mx.array_equal(dfdx, mx.array([1.0])))
        self.assertEqual(dfdx.dtype, mx.float32)

    def test_scatter_max_vjp(self):
        def fun(src, updates):
            x = src.at[1].maximum(updates)
            return x

        cotan = mx.array([4.0, 5.0, 6.0])
        _, vjps = mx.vjp(fun, [mx.array([1.0, 2.0, 3.0]), mx.array([[3.0]])], [cotan])
        mx.eval(vjps)

        # Update larger than value
        self.assertTrue(mx.allclose(vjps[0], mx.array([4.0, 0.0, 6.0])))
        self.assertTrue(mx.allclose(vjps[1], mx.array([5.0])))

        cotan = mx.array([[4.0], [5.0], [6.0]])
        _, vjps = mx.vjp(
            fun, [mx.array([[1.0], [2.0], [3.0]]), mx.array([[[2.0]]])], [cotan]
        )
        mx.eval(vjps)

        # Update and value are equal
        self.assertTrue(mx.allclose(vjps[0], mx.array([[4.0], [5.0], [6.0]])))
        self.assertTrue(mx.allclose(vjps[1], mx.array([[[5.0]]])))

    def test_scatter_min_vjp(self):
        def fun(src, updates):
            x = src.at[1].minimum(updates)
            return x

        cotan = mx.array([4.0, 5.0, 6.0])
        _, vjps = mx.vjp(fun, [mx.array([1.0, 2.0, 3.0]), mx.array([[3.0]])], [cotan])
        mx.eval(vjps)

        # Update larger than value
        self.assertTrue(mx.allclose(vjps[0], mx.array([4.0, 5.0, 6.0])))
        self.assertTrue(mx.allclose(vjps[1], mx.array([0.0])))

        cotan = mx.array([[4.0], [5.0], [6.0]])
        _, vjps = mx.vjp(
            fun, [mx.array([[1.0], [2.0], [3.0]]), mx.array([[[2.0]]])], [cotan]
        )
        mx.eval(vjps)

        # Update and value are equal
        self.assertTrue(mx.allclose(vjps[0], mx.array([[4.0], [5.0], [6.0]])))
        self.assertTrue(mx.allclose(vjps[1], mx.array([[[5.0]]])))

    def test_split_against_slice(self):
        def f_split(x):
            a, _, b = x.split(3, -1)
            return (a * b).sum()

        def f_slice(x):
            step = x.shape[-1] // 3
            a = x[..., :step]
            b = x[..., -step:]
            return (a * b).sum()

        x = mx.random.uniform(shape=(100, 300))
        mx.eval(x)

        df1 = mx.grad(f_split)
        df2 = mx.grad(f_slice)

        self.assertTrue(mx.allclose(df1(x), df2(x)))

    def test_vjp_types(self):
        def fun(x):
            return x

        for t in [mx.float16, mx.bfloat16, mx.float32]:
            out = mx.grad(fun)(mx.array(1.0, t))
            self.assertEqual(out.dtype, t)

        def fun(x):
            return x.sum()

        for t in [mx.float16, mx.bfloat16, mx.float32]:
            out = mx.grad(fun)(mx.array(1.0, t))
            self.assertEqual(out.dtype, t)

        def fun(x, y):
            return (x + y).sum()

        for t in [mx.float16, mx.bfloat16, mx.float32]:
            out = mx.grad(fun)(mx.array(1.0, t), mx.array(1.0, t))
            self.assertEqual(out.dtype, t)

    def test_power_grad(self):
        x = mx.array(0.0)
        g = mx.grad(lambda x: x**2)(x)
        self.assertEqual(g.item(), 0.0)

        x = mx.array(0.0)
        g = mx.grad(lambda x: x**1.5)(x)
        self.assertEqual(g.item(), 0.0)

        x = mx.array(2.0)
        g = mx.grad(lambda x: x**2)(x)
        self.assertAlmostEqual(g.item(), 4.0)

    def test_eval_in_grad(self):
        arr = mx.array([1.0])
        cotan = mx.array([1.0, 1.0])
        y = mx.array([2.0, 2.0])

        def func(x):
            x = x + y
            cond = x < 1
            cond.tolist()
            return x**2

        _, vjps = mx.vjp(func, (arr,), (cotan,))
        self.assertEqual(vjps[0].item(), 12.0)

        def func(x):
            x = x + mx.array([1.0, 1.0])
            mx.eval(x)
            return x**2

        _, vjps = mx.vjp(func, (arr,), (cotan,))
        self.assertEqual(vjps[0].item(), 8.0)

    def test_power_grad(self):
        def fun(x, y):
            res = x - y
            return res**x

        grad = mx.grad(fun)(mx.array(1.0), mx.array(1.0))
        self.assertEqual(grad.item(), 1.0)

    def test_cumprod_grad(self):
        def fun(y):
            return mx.cumprod(y).sum()

        y = mx.array([2.0, 1.0, 2.0, 2.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([20.0, 38.0, 18.0, 16.0, 8.0])
        self.assertTrue(mx.allclose(out, expected))

        y = mx.array([2.0, 0.0, 2.0, 2.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([1.0, 38.0, 0.0, 0.0, 0.0])
        self.assertTrue(mx.allclose(out, expected))

        y = mx.array([2.0, 0.0, 2.0, 0.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([1.0, 6.0, 0.0, 0.0, 0.0])
        self.assertTrue(mx.allclose(out, expected))

        def fun(y):
            return mx.cumprod(y, inclusive=False).sum()

        y = mx.array([2.0, 1.0, 2.0, 2.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([8.0, 14.0, 6.0, 4.0, 0.0])
        self.assertTrue(mx.allclose(out, expected))

        y = mx.array([2.0, 0.0, 2.0, 2.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([1.0, 14.0, 0.0, 0.0, 0.0])
        self.assertTrue(mx.allclose(out, expected))

        y = mx.array([2.0, 0.0, 2.0, 0.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([1.0, 6.0, 0.0, 0.0, 0.0])
        self.assertTrue(mx.allclose(out, expected))

        def fun(y):
            return mx.cumprod(y, inclusive=False, reverse=True).sum()

        y = mx.array([2.0, 1.0, 2.0, 2.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([0.0, 12.0, 12.0, 15.0, 11.0])
        self.assertTrue(mx.allclose(out, expected))

        y = mx.array([2.0, 0.0, 2.0, 2.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([0.0, 12.0, 6.0, 9.0, 7.0])
        self.assertTrue(mx.allclose(out, expected))

        y = mx.array([2.0, 0.0, 2.0, 0.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([0.0, 0.0, 0.0, 9.0, 1.0])
        self.assertTrue(mx.allclose(out, expected))

        def fun(y):
            return mx.cumprod(y, reverse=True).sum()

        y = mx.array([2.0, 1.0, 2.0, 2.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([12.0, 36.0, 24.0, 27.0, 19.0])
        self.assertTrue(mx.allclose(out, expected))

        y = mx.array([2.0, 0.0, 2.0, 2.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([0.0, 36.0, 6.0, 9.0, 7.0])
        self.assertTrue(mx.allclose(out, expected))

        y = mx.array([2.0, 0.0, 2.0, 0.0, 3.0])
        out = mx.grad(fun)(y)
        expected = mx.array([0.0, 0.0, 0.0, 9.0, 1.0])
        self.assertTrue(mx.allclose(out, expected))

    def test_topk_grad(self):
        a = mx.array([[1, 2, 6, 4, 5], [9, 5, 6, 7, 8]], mx.float32)

        def fun(x):
            return mx.topk(x, 2)

        out = mx.vjp(fun, (a,), (mx.ones((2, 2)),))[1][0]
        expected = mx.array([[0, 0, 1, 0, 1], [1, 0, 0, 0, 1]], mx.float32)
        self.assertTrue(mx.array_equal(out, expected))

    def test_custom_function(self):
        # Make a custom function
        my_exp = mx.custom_function(mx.exp)

        # Ensure everything works
        dy = mx.grad(my_exp)(mx.array(1.0))
        self.assertTrue(mx.allclose(dy, mx.exp(mx.array(1.0))))
        (ex,), (dex,) = mx.jvp(my_exp, [mx.array(1.0)], [mx.array(1.0)])
        self.assertTrue(mx.allclose(dex, mx.exp(mx.array(1.0))))
        self.assertTrue(mx.allclose(ex, dex))
        ex = mx.vmap(my_exp)(mx.ones(10))
        self.assertTrue(mx.allclose(ex, mx.exp(mx.ones(10))))

        # Ensure that the vjp is being overriden but everything else still
        # works.
        @my_exp.vjp
        def my_exp_vjp(x, dx, ex):
            return mx.ones_like(x) * 42

        dy = mx.grad(my_exp)(mx.array(1.0))
        self.assertTrue(mx.allclose(dy, mx.array(42.0)))
        (ex,), (dex,) = mx.jvp(my_exp, [mx.array(1.0)], [mx.array(1.0)])
        self.assertTrue(mx.allclose(dex, mx.exp(mx.array(1.0))))
        self.assertTrue(mx.allclose(ex, dex))
        ex = mx.vmap(my_exp)(mx.ones(10))
        self.assertTrue(mx.allclose(ex, mx.exp(mx.ones(10))))

        # Ensure that setting the jvp and vmap also works.
        @my_exp.jvp
        def my_exp_jvp(x, dx):
            return mx.ones_like(x) * 7 * dx

        @my_exp.vmap
        def my_exp_vmap(x, axis):
            return mx.ones_like(x) * 3, axis

        dy = mx.grad(my_exp)(mx.array(1.0))
        self.assertTrue(mx.allclose(dy, mx.array(42.0)))
        (ex,), (dex,) = mx.jvp(my_exp, [mx.array(1.0)], [mx.array(1.0)])
        self.assertTrue(mx.allclose(dex, mx.array(7.0)))
        self.assertTrue(mx.allclose(ex, mx.exp(mx.array(1.0))))
        ex = mx.vmap(my_exp)(mx.ones(10))
        self.assertTrue(mx.allclose(ex, 3 * mx.ones(10)))

        # Test pytrees
        @mx.custom_function
        def my_double(params):
            return {"out": 2 * params["x"] * params["y"]}

        dy = mx.grad(lambda p: my_double(p)["out"].sum())(
            {"x": mx.ones(2), "y": mx.ones(2)}
        )
        self.assertTrue(mx.allclose(dy["x"], mx.ones(2) * 2))
        self.assertTrue(mx.allclose(dy["y"], mx.ones(2) * 2))

        @my_double.vjp
        def random_grads(primals, cotangents, outputs):
            return {"x": mx.zeros_like(primals["x"]), "y": mx.ones_like(primals["y"])}

        dy = mx.grad(lambda p: my_double(p)["out"].sum())(
            {"x": mx.ones(2), "y": mx.ones(2)}
        )
        self.assertTrue(mx.allclose(dy["x"], mx.zeros(2)))
        self.assertTrue(mx.allclose(dy["y"], mx.ones(2)))

        def outer_f(a, b):
            return my_double({"x": a, "y": b})["out"]

        inputs = [mx.random.normal(shape=(2,)) for i in range(2)]
        tans = [mx.random.normal(shape=(2,)) for i in range(2)]
        out1, dout1 = mx.jvp(outer_f, inputs, tans)

        @my_double.jvp
        def random_grads(primals, tangents):
            return {
                "out": 2 * primals["x"] * tangents["y"]
                + 2 * primals["y"] * tangents["x"]
                + 1
            }

        out2, dout2 = mx.jvp(outer_f, inputs, tans)
        self.assertTrue(mx.allclose(out1[0], out2[0]))
        self.assertTrue(mx.allclose(dout1[0] + 1, dout2[0]))

    def test_complex_vjps(self):
        def fun(x):
            return (2.0 * mx.real(x)).sum()

        x = mx.array([0.0 + 1j, 1.0 + 0.0j, 0.5 + 0.5j])
        dfdx = mx.grad(fun)(x)
        self.assertTrue(mx.allclose(dfdx, 2 * mx.ones_like(x)))

        def fun(x):
            return (2.0 * mx.imag(x)).sum()

        x = mx.array([0.0 + 1j, 1.0 + 0.0j, 0.5 + 0.5j])
        dfdx = mx.grad(fun)(x)
        self.assertTrue(mx.allclose(dfdx, 2j * mx.ones_like(x)))

    def test_flatten_unflatten_vjps(self):
        def fun(x):
            y = mx.unflatten(x, 0, (2, 2))
            return y.sum()

        x = mx.zeros((4, 8))
        self.assertEqual(mx.grad(fun)(x).shape, (4, 8))

        def fun(x):
            y = mx.flatten(x, 0, 2)
            return y.sum()

        x = mx.zeros((2, 4, 8))
        self.assertEqual(mx.grad(fun)(x).shape, (2, 4, 8))

    def test_concatenate_vjps(self):
        def fun(x, y):
            return mx.concatenate([x, y])

        x = mx.array([1, 2, 3], mx.float32)
        y = mx.array([1, 2, 3], mx.float16)
        grads = mx.vjp(fun, (x, y), (mx.ones((6,)),))[1]
        self.assertTrue(mx.allclose(grads[0], mx.ones(3)))
        self.assertTrue(mx.allclose(grads[1], mx.ones(3)))
        self.assertEqual(grads[0].dtype, mx.float32)
        self.assertEqual(grads[1].dtype, mx.float16)

    def test_matmul_jvps(self):
        a = mx.random.uniform(shape=(4, 4))
        b = mx.random.uniform(shape=(4, 4))
        c = mx.random.uniform(shape=(4, 4))
        d = mx.random.uniform(shape=(4, 4))

        _, tangent = mx.jvp(lambda a: a @ b, (a,), (c,))
        self.assertTrue(mx.allclose(tangent[0], c @ b))

        _, tangent = mx.jvp(lambda b: a @ b, (b,), (d,))
        self.assertTrue(mx.allclose(tangent[0], a @ d))

        _, tangent = mx.jvp(lambda a, b: a @ b, (a, b), (c, d))
        self.assertTrue(mx.allclose(tangent[0], a @ d + c @ b))

        x = mx.random.uniform(shape=(4, 4))
        y = mx.random.uniform(shape=(4, 4))
        z = mx.random.uniform(shape=(4, 4))

        _, (tangent,) = mx.jvp(lambda a, b, c: a @ b + c, (a, b, c), (x, y, z))
        _, (expected,) = mx.jvp(lambda a, b, c: mx.addmm(c, a, b), (a, b, c), (x, y, z))
        self.assertTrue(mx.allclose(tangent, expected))

        _, (tangent,) = mx.jvp(lambda a, c: a @ b + c, (a, c), (x, z))
        _, (expected,) = mx.jvp(lambda a, c: mx.addmm(c, a, b), (a, c), (x, z))
        self.assertTrue(mx.allclose(tangent, expected))

        _, (tangent,) = mx.jvp(lambda b, c: a @ b + c, (b, c), (y, z))
        _, (expected,) = mx.jvp(lambda b, c: mx.addmm(c, a, b), (b, c), (y, z))
        self.assertTrue(mx.allclose(tangent, expected))

        _, (tangent,) = mx.jvp(lambda c: a @ b + c, (c,), (z,))
        _, (expected,) = mx.jvp(lambda c: mx.addmm(c, a, b), (c,), (z,))
        self.assertTrue(mx.allclose(tangent, expected))

    def test_put_along_axis_grads(self):
        a = mx.zeros((5, 1))
        b = mx.ones((2, 1))

        def fun(a, b):
            idx = mx.array([[0], [3]])
            return mx.put_along_axis(a, idx, b, axis=0)

        # Test VJP
        cotan = mx.full((5, 1), 2.0)
        _, (da, db) = mx.vjp(fun, (a, b), (cotan,))
        expected_da = mx.array([0.0, 2.0, 2.0, 0.0, 2.0])[:, None]
        expected_db = mx.array([2.0, 2.0])[:, None]
        self.assertTrue(mx.allclose(expected_da, da))
        self.assertTrue(mx.allclose(expected_db, db))

        # Test JVP
        tan_a = mx.full((5, 1), 2.0)
        tan_b = mx.full((2, 1), 3.0)
        _, (jout,) = mx.jvp(fun, (a, b), (tan_a, tan_b))
        expected = mx.array([3.0, 2.0, 2.0, 3.0, 2.0])[:, None]
        self.assertTrue(mx.allclose(expected, jout))

        def fun(a):
            idx = mx.array([[0], [3]])
            return mx.put_along_axis(a, idx, b, axis=0)

        _, (jout,) = mx.jvp(fun, (a,), (tan_a,))
        expected = mx.array([0.0, 2.0, 2.0, 0.0, 2.0])[:, None]
        self.assertTrue(mx.allclose(expected, jout))

    def test_slice_grads(self):
        # Slice
        def fun(a):
            return a[5:-6:-1]

        a = mx.ones(shape=(5,))
        cotan = mx.random.uniform(shape=(5,))
        _, (grad,) = mx.vjp(fun, (a,), (cotan,))
        self.assertTrue(mx.allclose(grad, cotan[::-1]))

        tan = mx.random.uniform(shape=(5,))
        mx.eval(tan)
        _, (grad,) = mx.jvp(fun, (a,), (tan,))
        self.assertTrue(mx.allclose(grad, tan[::-1]))

        # Slice update
        def fun(a, b):
            a[4:-5:-2] = b
            return a

        a = mx.ones(shape=(4,))
        b = mx.zeros(shape=(2,))

        cotan = mx.random.uniform(shape=(4,))
        _, (grad_a, grad_b) = mx.vjp(fun, (a, b), (cotan,))
        expected_a = mx.array(cotan)
        expected_a[1::2] = 0.0
        self.assertTrue(mx.allclose(grad_a, expected_a))
        self.assertTrue(mx.allclose(grad_b, cotan[4:-5:-2]))

        tan_a = mx.random.uniform(shape=(4,))
        tan_b = mx.random.uniform(shape=(2,))
        _, (grad,) = mx.jvp(fun, (a, b), (tan_a, tan_b))
        expected = tan_a
        expected[4:-5:-2] = tan_b
        self.assertTrue(mx.allclose(grad, expected))

    def test_leaks(self):
        for transform in [
            mx.grad,
            mx.value_and_grad,
            mx.custom_function,
            mx.checkpoint,
        ]:
            mx.synchronize()
            mem_pre = mx.get_active_memory()

            def outer():
                d = {}

                def f(x):
                    return d["x"]

                d["f"] = transform(f)
                d["x"] = mx.array([0] * 1000)

            for _ in range(5):
                outer()
                gc.collect()
            mem_post = mx.get_active_memory()
            self.assertEqual(mem_pre, mem_post)

    def test_grad_with_copies(self):
        a = mx.array(2.0)
        arrays = [a, a, a]

        def fun(arrays):
            return arrays[0] + arrays[2]

        grads = mx.grad(fun)(arrays)
        self.assertEqual(grads[0].item(), 1.0)
        self.assertEqual(grads[2].item(), 1.0)

    def test_grad_ids_pre_post(self):
        def fun(arrs):
            return arrs[0]

        arrs = [mx.array(1.0)]
        init_id = id(arrs[0])
        mx.grad(fun)(arrs)
        self.assertEqual(init_id, id(arrs[0]))

    def test_grad_with_inplace_update(self):
        def loss_fn(model):
            model[1] = mx.array(2.0)
            return model[0]

        model = [
            mx.array(0.0),
            mx.array(1.0),
        ]

        grad_fn = mx.grad(loss_fn)
        grad_fn(model)
        self.assertEqual(model[1].item(), 2.0)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
