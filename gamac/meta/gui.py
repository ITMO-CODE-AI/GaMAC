import os
import time

from gamac.src.meta.storage import traverse_data, PARTITIONS_TO_ESTIMATE

ALT_PUSH, ALT_PULL = "PUSH", "PULL"
ACCESSOR_IDX = 0


class ComparisonContext:
    def __init__(self, data_path, sorted_indices):
        self.alt_push_idx = len(sorted_indices)
        self.alt_pull_idx = self.alt_push_idx - 1

        self.sorted_indices = sorted_indices
        self.data_path = data_path

        self.alternative = ALT_PUSH
        self.t_start = time.time()

    def swap(self):
        self.alternative = ALT_PUSH if self.alternative == ALT_PULL else ALT_PULL

    def shift(self):
        self.alt_pull_idx -= 1
        assert self.alt_pull_idx >= 0

    def rendered_image_idx(self):
        return self.alt_push_idx if self.alternative == ALT_PUSH else self.sorted_indices[self.alt_pull_idx]

    def label(self):
        idx = self.rendered_image_idx()
        return f'{self.alternative} [{self.data_path}, image #{idx}]'

    def colour(self):
        return 'green' if self.alternative == ALT_PUSH else 'blue'


class DataContext:
    def __init__(self, data_path):
        self.data_path, self._comp_ctx = data_path, None
        self.images = [self._photo_image(idx) for idx in range(PARTITIONS_TO_ESTIMATE)]

        self._accessor_path = f'data/{data_path}/accessor-{ACCESSOR_IDX}.txt'
        if os.path.exists(self._accessor_path):
            raise FileExistsError(f"Data {data_path} has been already estimated by accessor {ACCESSOR_IDX}")

        self._sorted_indices, self._comparisons = [0], []

    def comp_ctx(self) -> ComparisonContext:
        return self._comp_ctx

    def next_comp_ctx(self):
        if self.has_next():
            self._comp_ctx = ComparisonContext(self.data_path, self._sorted_indices)

    def has_next(self):
        return len(self._sorted_indices) < PARTITIONS_TO_ESTIMATE

    def choose(self):
        comp_ctx = self.comp_ctx()
        self._save_comparison(comp_ctx)
        self._save_index_or_shift(comp_ctx)

    def _save_comparison(self, comp_ctx):
        t_estimate = time.time() - comp_ctx.t_start
        t_estimate = float(int(t_estimate * 10) / 10)
        push_idx, pull_idx = comp_ctx.alt_push_idx, comp_ctx.alt_pull_idx
        if comp_ctx.alternative == ALT_PUSH:
            self._comparisons.append((push_idx, pull_idx, t_estimate))
        else:
            self._comparisons.append((pull_idx, push_idx, t_estimate))

    def _save_index_or_shift(self, comp_ctx):
        if comp_ctx.alternative == ALT_PUSH:
            insert_at = comp_ctx.alt_pull_idx + 1
            self._sorted_indices.insert(insert_at, comp_ctx.alt_push_idx)
            self.next_comp_ctx()
        elif comp_ctx.alternative == ALT_PULL and comp_ctx.alt_pull_idx == 0:
            self._sorted_indices.insert(0, comp_ctx.alt_push_idx)
            self.next_comp_ctx()
        else:
            comp_ctx.shift()

    def persist(self):
        with open(self._accessor_path, 'w') as fp:
            fp.writelines([
                self._sorted_indices.__str__(),
                "\n",
                self._comparisons.__str__()
            ])

    def _photo_image(self, idx):
        return PhotoImage(file=f'data/{self.data_path}/img-{idx}.png')


class Application:
    def __init__(self, data_for_estimation):
        self._all_data = data_for_estimation
        self._current_idx, self._data_ctx = -1, None

    def data_ctx(self) -> DataContext:
        return self._data_ctx

    def next_data_ctx(self):
        if self._current_idx < len(self._all_data):
            self._current_idx += 1
            data_path = self._all_data[self._current_idx]
            self._data_ctx = DataContext(data_path)
            self._data_ctx.next_comp_ctx()
        else:
            raise AttributeError('Congratulations! You have estimated all available data')


def is_estimated(data_path):
    content = os.listdir(f'data/{data_path}')
    return f'accessor-{ACCESSOR_IDX}.txt' in content


if __name__ == '__main__':
    from tkinter import *

    root = Tk()
    root.title(f"GAMaC Assessment [Accessor #{ACCESSOR_IDX}]")
    root.geometry("3200x2400")

    data_for_estimation = [
        data_path for data_path, estimated in traverse_data(is_estimated).items() if not estimated
    ]
    application = Application(data_for_estimation)

    header = Label(font=("Arial", 20))
    header.pack()

    image = Label()
    image.pack(expand=True, anchor=CENTER)


    def update_ui():
        data_ctx = application.data_ctx()
        comp_ctx = data_ctx.comp_ctx()

        header.config(
            text=comp_ctx.label(),
            foreground=comp_ctx.colour()
        )

        image_idx = comp_ctx.rendered_image_idx()
        image.config(
            highlightbackground=comp_ctx.colour(),
            highlightthickness=4,
            image=data_ctx.images[image_idx]
        )


    def get_ready_ui():
        header.config(
            text=f'Press <Enter> to start next assessment',
            foreground='red'
        )
        image.config(
            highlightbackground='red',
            highlightthickness=4,
        )
        root.update_idletasks()


    def launch_next(_):
        data_ctx = application.data_ctx()
        if data_ctx is None or not data_ctx.has_next():
            application.next_data_ctx()
            update_ui()


    def swap(_):
        application.data_ctx().comp_ctx().swap()
        update_ui()


    def choose(_):
        data_ctx = application.data_ctx()
        data_ctx.choose()
        if not data_ctx.has_next():
            data_ctx.persist()
            get_ready_ui()
        else:
            update_ui()


    root.bind('<KeyPress-Return>', launch_next)
    root.bind('<KeyPress-Tab>', swap)
    root.bind('<space>', choose)

    get_ready_ui()
    root.mainloop()
